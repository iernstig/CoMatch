import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from src.config.default import get_cfg_defaults
from src.loftr import LoFTR
from src.loftr.loftr_module.linear_attention import Attention
from src.utils.misc import lower_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run a simple two-image CoMatch inference smoke test.")
    parser.add_argument(
        "--image0",
        type=Path,
        default=Path("/mnt/qvo/personal/isak/data/henrik-bristle-img/001.png"),
        # default=Path("assets/phototourism_sample_images/london_bridge_19481797_2295892421.jpg"),
        help="Path to the first image.",
    )
    parser.add_argument(
        "--image1",
        type=Path,
        default=Path("/mnt/qvo/personal/isak/data/henrik-bristle-img/002.png"),
        # default=Path("assets/phototourism_sample_images/london_bridge_49190386_5209386933.jpg"),
        help="Path to the second image.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=Path("weights/comatch_outdoor.ckpt"),
        help="Path to the checkpoint.",
    )
    parser.add_argument(
        "--main-cfg-path",
        type=Path,
        default=Path("configs/loftr/comatch_full.py"),
        help="Path to the main model config.",
    )
    parser.add_argument(
        "--long-side",
        type=int,
        default=1152,
        help="Resize the longer image side to this value before rounding down to a multiple of 32.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/test_inference"),
        help="Directory for covisibility visualizations.",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=100,
        help="Maximum number of filtered matches to visualize.",
    )
    parser.add_argument(
        "--magsac-threshold",
        type=float,
        default=1.5,
        help="Reprojection threshold for OpenCV USAC_MAGSAC filtering.",
    )
    parser.add_argument(
        "--magsac-confidence",
        type=float,
        default=0.999,
        help="Confidence level for OpenCV USAC_MAGSAC filtering.",
    )
    parser.add_argument(
        "--magsac-max-iters",
        type=int,
        default=10000,
        help="Maximum iterations for OpenCV USAC_MAGSAC filtering.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=0,
        help="Number of warmup iterations before timing. Default is 0 for a single direct run path.",
    )
    parser.add_argument(
        "--timed-iters",
        type=int,
        default=50,
        help="Number of timed iterations used to compute average latency.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable mixed precision during inference timing and forward passes.",
    )
    parser.add_argument(
        "--coarse-softmax-mode",
        choices=["softmax", "log_softmax"],
        default="log_softmax",
        help="Dual-softmax implementation used inside coarse matching.",
    )
    parser.add_argument(
        "--compile-coarse-matching",
        action="store_true",
        help="Compile the coarse matching module with torch.compile using dynamic shapes.",
    )
    return parser.parse_args()


def load_gray_image(path: Path, long_side: int):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    height, width = image.shape
    crop_size = min(height, width)
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    image = image[top:top + crop_size, left:left + crop_size]
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    height, width = image.shape
    scale = long_side / max(height, width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))

    resized_width = max(32, (resized_width // 32) * 32)
    resized_height = max(32, (resized_height // 32) * 32)

    image = cv2.resize(image, (resized_width, resized_height))
    tensor = torch.from_numpy(image).float()[None][None] / 255.0
    return image, tensor


def save_covis(gray_image: np.ndarray, score_map: torch.Tensor, output_path: Path):
    score = score_map.detach().float().cpu().numpy().astype(np.float32, copy=False)
    score = cv2.resize(score, (gray_image.shape[1], gray_image.shape[0]))
    score = score - score.min()
    if score.max() > 0:
        score = score / score.max()

    masked = (gray_image.astype(np.float32) * score).clip(0, 255).astype(np.uint8)
    heat = cv2.applyColorMap((score * 255).astype(np.uint8), cv2.COLORMAP_JET)
    gray_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(gray_bgr, 0.55, heat, 0.45, 0.0)

    cv2.imwrite(str(output_path), overlay)
    cv2.imwrite(str(output_path.with_stem(f"{output_path.stem}_masked")), masked)
    return overlay, masked


def confidence_colormap(confidence: torch.Tensor):
    conf = confidence.detach().float().cpu().numpy().astype(np.float32, copy=False)
    if conf.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    conf = conf - conf.min()
    if conf.max() > 0:
        conf = conf / conf.max()
    colors = cv2.applyColorMap((conf * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)
    colors = colors.reshape(-1, 3)[:, ::-1].astype(np.float32) / 255.0
    alpha = np.full((colors.shape[0], 1), 0.85, dtype=np.float32)
    colors = np.concatenate([colors, alpha], axis=1)
    return colors


def save_matching_figure_native(img0, img1, mkpts0, mkpts1, color, text, output_path, dpi=200):
    total_width = img0.shape[1] + img1.shape[1]
    max_height = max(img0.shape[0], img1.shape[0])
    figsize = (total_width / dpi, max_height / dpi)

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    axes[0].imshow(img0)
    axes[1].imshow(img1)

    for axis in axes:
        axis.get_yaxis().set_ticks([])
        axis.get_xaxis().set_ticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.02)

    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        trans_figure = fig.transFigure.inverted()
        fig_pts0 = trans_figure.transform(axes[0].transData.transform(mkpts0))
        fig_pts1 = trans_figure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [
            Line2D(
                (fig_pts0[index, 0], fig_pts1[index, 0]),
                (fig_pts0[index, 1], fig_pts1[index, 1]),
                transform=fig.transFigure,
                c=color[index],
                linewidth=1.2,
            )
            for index in range(len(mkpts0))
        ]
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=10)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=10)

    txt_color = "k" if img0[:100, :200].mean() > 200 else "w"
    fig.text(0.01, 0.99, "\n".join(text), fontsize=16, va="top", ha="left", color=txt_color)
    fig.savefig(str(output_path), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def filter_matches_magsac(mkpts0, mkpts1, mconf, threshold, confidence, max_iters):
    if mkpts0.shape[0] < 8:
        return mkpts0, mkpts1, mconf, None, None

    method = getattr(cv2, "USAC_MAGSAC", None)
    if method is None:
        raise RuntimeError("This OpenCV build does not expose cv2.USAC_MAGSAC.")

    fundamental, inlier_mask = cv2.findFundamentalMat(
        mkpts0,
        mkpts1,
        method=method,
        ransacReprojThreshold=threshold,
        confidence=confidence,
        maxIters=max_iters,
    )

    if inlier_mask is None:
        return mkpts0[:0], mkpts1[:0], mconf[:0], fundamental, None

    inlier_mask = inlier_mask.reshape(-1).astype(bool)
    return mkpts0[inlier_mask], mkpts1[inlier_mask], mconf[inlier_mask], fundamental, inlier_mask


def select_match_subset(mkpts0, mkpts1, mconf, max_matches):
    if max_matches <= 0 or mkpts0.shape[0] <= max_matches:
        return mkpts0, mkpts1, mconf

    order = np.argsort(-mconf)
    keep = order[:max_matches]
    return mkpts0[keep], mkpts1[keep], mconf[keep]


def _autocast_context(device, enabled):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", enabled=enabled)


def _run_full_matcher(model, image0, image1, device, use_amp, coarse_softmax_mode):
    batch = {
        "image0": image0,
        "image1": image1,
        "_coarse_softmax_mode": coarse_softmax_mode,
    }
    with _autocast_context(device, use_amp):
        model(batch)
    return batch


def _run_coarse_path(model, image0, image1, device, use_amp):
    with _autocast_context(device, use_amp):
        features = model.backbone(torch.cat([image0, image1], dim=0))
        feat_c0, feat_c1 = features["feats_c"].split(1)
        feat_c0, feat_c1, scores0, scores1 = model.loftr_coarse(feat_c0, feat_c1)
    return feat_c0, feat_c1, scores0, scores1


def _sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_block(device, fn):
    _sync_if_needed(device)
    start_time = time.perf_counter()
    result = fn()
    _sync_if_needed(device)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return result, elapsed_ms


def _profile_forward_stages(model, image0, image1, device, use_amp, coarse_softmax_mode):
    with _autocast_context(device, use_amp):
        data = {
            "image0": image0,
            "image1": image1,
            "bs": image0.size(0),
            "hw0_i": image0.shape[2:],
            "hw1_i": image1.shape[2:],
            "_profile_coarse_matching": True,
            "_coarse_softmax_mode": coarse_softmax_mode,
        }

        total_start = time.perf_counter()

        ret_dict, backbone_ms = _time_block(
            device,
            lambda: model.backbone(torch.cat([image0, image1], dim=0)),
        )
        feats_c = ret_dict["feats_c"]
        data.update({
            "feats_x2": ret_dict["feats_x2"],
            "feats_x1": ret_dict["feats_x1"],
        })
        feat_c0, feat_c1 = feats_c.split(data["bs"])

        mul = model.config["resolution"][0] // model.config["resolution"][1]
        data.update({
            "hw0_c": feat_c0.shape[2:],
            "hw1_c": feat_c1.shape[2:],
            "hw0_f": [feat_c0.shape[2] * mul, feat_c0.shape[3] * mul],
            "hw1_f": [feat_c1.shape[2] * mul, feat_c1.shape[3] * mul],
        })

        (coarse_result, coarse_transformer_ms) = _time_block(
            device,
            lambda: model.loftr_coarse(feat_c0, feat_c1),
        )
        feat_c0, feat_c1, scores0, scores1 = coarse_result
        data.update({
            "matchability_score_list0": scores0,
            "matchability_score_list1": scores1,
        })

        feat_c0 = feat_c0.flatten(2).transpose(1, 2)
        feat_c1 = feat_c1.flatten(2).transpose(1, 2)

        (_, coarse_matching_ms) = _time_block(
            device,
            lambda: model.coarse_matching(feat_c0, feat_c1, data),
        )

        feat_c0 = feat_c0 / feat_c0.shape[-1] ** 0.5
        feat_c1 = feat_c1 / feat_c1.shape[-1] ** 0.5

        ((feat_f0_unfold, feat_f1_unfold), fine_preprocess_ms) = _time_block(
            device,
            lambda: model.fine_preprocess(feat_c0, feat_c1, data),
        )

        fine_transformer_ms = 0.0
        if feat_f0_unfold.size(0) != 0:
            ((feat_f0_unfold, feat_f1_unfold), fine_transformer_ms) = _time_block(
                device,
                lambda: model.loftr_fine(feat_f0_unfold, feat_f1_unfold),
            )

        (_, fine_matching_ms) = _time_block(
            device,
            lambda: model.fine_matching(feat_f0_unfold, feat_f1_unfold, data),
        )
        _sync_if_needed(device)
        total_ms = (time.perf_counter() - total_start) * 1000.0

    return data, {
        "backbone": backbone_ms,
        "coarse_transformer": coarse_transformer_ms,
        "coarse_matching": coarse_matching_ms,
        "coarse_matching_breakdown": data.get("coarse_matching_profile_ms", {}),
        "fine_preprocess": fine_preprocess_ms,
        "fine_transformer": fine_transformer_ms,
        "fine_matching": fine_matching_ms,
        "profiled_total": total_ms,
    }


def _attention_status(model):
    modules = [module for module in model.modules() if isinstance(module, Attention)]
    flash_flags = [module.flash for module in modules]
    return {
        "num_attention_modules": len(modules),
        "num_flash_enabled": sum(bool(flag) for flag in flash_flags),
        "all_flash_enabled": all(flash_flags) if flash_flags else False,
    }


def _maybe_compile_coarse_matching(model, enabled):
    if not enabled:
        return False, None
    if not hasattr(torch, "compile"):
        return False, "torch.compile unavailable"
    try:
        model.coarse_matching = torch.compile(
            model.coarse_matching,
            dynamic=True,
            mode="reduce-overhead",
            fullgraph=False,
        )
        return True, None
    except Exception as error:
        return False, str(error)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(args.main_cfg_path))
    cfg.LOFTR.COARSE.NPE = [832, 832, args.long_side, args.long_side]
    cfg.LOFTR.COARSE.NO_FLASH = False
    cfg.LOFTR.HALF = False
    cfg.LOFTR.MP = not args.disable_amp

    use_amp = device.type == "cuda" and cfg.LOFTR.MP

    model = LoFTR(config=lower_config(cfg)["loftr"]).to(device)
    state = torch.load(str(args.ckpt_path), map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    coarse_compile_enabled, coarse_compile_error = _maybe_compile_coarse_matching(model, args.compile_coarse_matching)
    attention_status = _attention_status(model)

    gray0, image0 = load_gray_image(args.image0, args.long_side)
    gray1, image1 = load_gray_image(args.image1, args.long_side)
    image0 = image0.to(device)
    image1 = image1.to(device)

    with torch.inference_mode():
        for _ in range(args.warmup_iters):
            _run_full_matcher(model, image0, image1, device, use_amp, args.coarse_softmax_mode)

        if device.type == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(args.timed_iters):
                _run_full_matcher(model, image0, image1, device, use_amp, args.coarse_softmax_mode)
            end_event.record()
            torch.cuda.synchronize()
            inference_ms = start_event.elapsed_time(end_event) / max(args.timed_iters, 1)
        else:
            start_time = time.perf_counter()
            for _ in range(args.timed_iters):
                _run_full_matcher(model, image0, image1, device, use_amp, args.coarse_softmax_mode)
            inference_ms = ((time.perf_counter() - start_time) * 1000.0) / max(args.timed_iters, 1)

        batch, stage_timings_ms = _profile_forward_stages(model, image0, image1, device, use_amp, args.coarse_softmax_mode)

    mkpts0_f = batch["mkpts0_f"].detach().cpu().numpy()
    mkpts1_f = batch["mkpts1_f"].detach().cpu().numpy()
    mconf = batch["mconf"].detach().cpu().numpy()

    (filtered_result, magsac_ms) = _time_block(
        device,
        lambda: filter_matches_magsac(
            mkpts0_f,
            mkpts1_f,
            mconf,
            threshold=args.magsac_threshold,
            confidence=args.magsac_confidence,
            max_iters=args.magsac_max_iters,
        ),
    )
    filtered_mkpts0, filtered_mkpts1, filtered_mconf, fundamental, inlier_mask = filtered_result
    subset_mkpts0, subset_mkpts1, subset_mconf = select_match_subset(
        filtered_mkpts0,
        filtered_mkpts1,
        filtered_mconf,
        args.max_matches,
    )

    score_maps0 = batch["matchability_score_list0"]
    score_maps1 = batch["matchability_score_list1"]

    print("input resolution image0:", f"{gray0.shape[1]}x{gray0.shape[0]}")
    print("input resolution image1:", f"{gray1.shape[1]}x{gray1.shape[0]}")
    print("mixed precision:", use_amp)
    print("flash attention status:", attention_status)
    print("coarse softmax mode:", args.coarse_softmax_mode)
    print("compile coarse matching:", coarse_compile_enabled)
    if coarse_compile_error is not None:
        print("compile coarse matching error:", coarse_compile_error)
    print("warmup iterations:", args.warmup_iters)
    print("timed iterations:", args.timed_iters)
    print("average inference time (ms):", f"{inference_ms:.2f}")
    print("stage timings (ms):")
    print("  backbone:", f"{stage_timings_ms['backbone']:.2f}")
    print("  coarse transformer:", f"{stage_timings_ms['coarse_transformer']:.2f}")
    print("  coarse matching:", f"{stage_timings_ms['coarse_matching']:.2f}")
    coarse_matching_breakdown = stage_timings_ms["coarse_matching_breakdown"]
    if coarse_matching_breakdown:
        print("    similarity:", f"{coarse_matching_breakdown.get('similarity', 0.0):.2f}")
        print("    threshold:", f"{coarse_matching_breakdown.get('threshold', 0.0):.2f}")
        print("    reshape 5d:", f"{coarse_matching_breakdown.get('reshape_5d', 0.0):.2f}")
        print("    border mask:", f"{coarse_matching_breakdown.get('border_mask', 0.0):.2f}")
        print("    reshape 3d:", f"{coarse_matching_breakdown.get('reshape_3d', 0.0):.2f}")
        print("    row max:", f"{coarse_matching_breakdown.get('row_max', 0.0):.2f}")
        print("    col max:", f"{coarse_matching_breakdown.get('col_max', 0.0):.2f}")
        print("    mutual nearest:", f"{coarse_matching_breakdown.get('mutual_nearest', 0.0):.2f}")
        if "mask_fill" in coarse_matching_breakdown:
            print("    mask fill:", f"{coarse_matching_breakdown.get('mask_fill', 0.0):.2f}")
        if "softmax" in coarse_matching_breakdown:
            print("    softmax:", f"{coarse_matching_breakdown.get('softmax', 0.0):.2f}")
        print("    mask reduce:", f"{coarse_matching_breakdown.get('mask_reduce', 0.0):.2f}")
        print("    where:", f"{coarse_matching_breakdown.get('where', 0.0):.2f}")
        print("    gather j ids:", f"{coarse_matching_breakdown.get('gather_j_ids', 0.0):.2f}")
        print("    gather conf:", f"{coarse_matching_breakdown.get('gather_conf', 0.0):.2f}")
        print("    decode mkpts0:", f"{coarse_matching_breakdown.get('decode_mkpts0', 0.0):.2f}")
        print("    decode mkpts1:", f"{coarse_matching_breakdown.get('decode_mkpts1', 0.0):.2f}")
        print("    extract matches:", f"{coarse_matching_breakdown.get('extract_matches', 0.0):.2f}")
    print("  fine preprocess:", f"{stage_timings_ms['fine_preprocess']:.2f}")
    print("  fine transformer:", f"{stage_timings_ms['fine_transformer']:.2f}")
    print("  fine matching:", f"{stage_timings_ms['fine_matching']:.2f}")
    print("  profiled total:", f"{stage_timings_ms['profiled_total']:.2f}")
    print("num covis maps:", len(score_maps0), len(score_maps1))
    print("num matches:", int(batch["mconf"].shape[0]))
    print("num MAGSAC inliers:", int(filtered_mconf.shape[0]))
    print("num visualized matches:", int(subset_mconf.shape[0]))
    print("post-processing timings (ms):")
    print("  MAGSAC:", f"{magsac_ms:.2f}")
    if fundamental is not None:
        print("estimated fundamental matrix:\n", fundamental)

    covis0_path = args.output_dir / "covis0.jpg"
    covis1_path = args.output_dir / "covis1.jpg"
    (covis0_result, covis0_ms) = _time_block(
        device,
        lambda: save_covis(gray0, score_maps0[-1][0, 0], covis0_path),
    )
    (covis1_result, covis1_ms) = _time_block(
        device,
        lambda: save_covis(gray1, score_maps1[-1][0, 0], covis1_path),
    )
    overlay0, _ = covis0_result
    overlay1, _ = covis1_result

    match_figure_path = args.output_dir / "covis_matches.jpg"
    (_, match_figure_ms) = _time_block(
        device,
        lambda: save_matching_figure_native(
            cv2.cvtColor(overlay0, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(overlay1, cv2.COLOR_BGR2RGB),
            subset_mkpts0,
            subset_mkpts1,
            confidence_colormap(torch.from_numpy(subset_mconf)),
            [
                f"#Fine bilateral matches {int(batch['mconf'].shape[0])}",
                f"#MAGSAC inliers {int(filtered_mconf.shape[0])}",
                f"#Visualized {int(subset_mconf.shape[0])}",
            ],
            match_figure_path,
        ),
    )

    print("  covis image 0:", f"{covis0_ms:.2f}")
    print("  covis image 1:", f"{covis1_ms:.2f}")
    print("  match figure:", f"{match_figure_ms:.2f}")

    visualization_ms = covis0_ms + covis1_ms + match_figure_ms
    postprocess_ms = magsac_ms
    end_to_end_with_vis_ms = stage_timings_ms['profiled_total'] + postprocess_ms + visualization_ms
    print("timing summary (ms):")
    print("  model only:", f"{stage_timings_ms['profiled_total']:.2f}")
    print("  post-process only:", f"{postprocess_ms:.2f}")
    print("  visualization only:", f"{visualization_ms:.2f}")
    print("  model + post + vis:", f"{end_to_end_with_vis_ms:.2f}")

    print("saved:", covis0_path, covis1_path, match_figure_path)


if __name__ == "__main__":
    main()