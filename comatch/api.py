from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

from src.config.default import get_cfg_defaults
from src.loftr import LoFTR
from src.utils.misc import lower_config


@dataclass(frozen=True)
class MatchResult:
    mkpts0: np.ndarray
    mkpts1: np.ndarray
    confidence: np.ndarray
    covis0: np.ndarray
    covis1: np.ndarray
    image0: np.ndarray
    image1: np.ndarray


def _default_config_path() -> str:
    return str(resources.files("configs.loftr").joinpath("comatch_full.py"))


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _normalize_image_array(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3:
        if image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] == 1:
            gray = image[..., 0]
        else:
            gray = image[..., :3].astype(np.float32).mean(axis=-1)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if gray.dtype == np.uint8:
        return gray

    gray = gray.astype(np.float32)
    if gray.size == 0:
        raise ValueError("Image cannot be empty")
    if gray.max() <= 1.0:
        gray = gray * 255.0
    return np.clip(gray, 0.0, 255.0).astype(np.uint8)


def _image_to_grayscale(image: Any) -> np.ndarray:
    if isinstance(image, (str, Path)):
        gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        return gray

    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"), dtype=np.uint8)

    if isinstance(image, torch.Tensor):
        return _normalize_image_array(image.detach().cpu().numpy())

    if isinstance(image, np.ndarray):
        return _normalize_image_array(image)

    raise TypeError(
        "Unsupported image type. Expected a path, PIL image, numpy array, or torch tensor."
    )


def _prepare_grayscale_image(
    image: Any,
    long_side: int,
) -> tuple[np.ndarray, torch.Tensor]:
    gray = _image_to_grayscale(image)

    height, width = gray.shape
    scale = long_side / max(height, width)
    resized_width = max(32, (int(round(width * scale)) // 32) * 32)
    resized_height = max(32, (int(round(height * scale)) // 32) * 32)
    resized = cv2.resize(gray, (resized_width, resized_height))
    tensor = torch.from_numpy(resized).float()[None, None] / 255.0
    return resized, tensor


def _resize_covisibility_map(score_map: torch.Tensor, image_shape: tuple[int, int]) -> np.ndarray:
    covis = score_map.detach().float().cpu().numpy().astype(np.float32, copy=False)
    covis = cv2.resize(covis, (image_shape[1], image_shape[0]))
    covis = covis - covis.min()
    max_value = covis.max()
    if max_value > 0:
        covis = covis / max_value
    return covis


class CoMatchMatcher:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device | None = None,
        long_side: int = 1152,
        use_amp: bool | None = None,
        config_path: str | Path | None = None,
        enable_flash: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = str(config_path) if config_path is not None else _default_config_path()
        self.device = _resolve_device(device)
        self.long_side = long_side

        cfg = get_cfg_defaults()
        cfg.merge_from_file(self.config_path)
        cfg.LOFTR.COARSE.NPE = [832, 832, long_side, long_side]
        cfg.LOFTR.COARSE.NO_FLASH = not enable_flash
        cfg.LOFTR.HALF = False

        if use_amp is None:
            use_amp = self.device.type == "cuda"
        cfg.LOFTR.MP = bool(use_amp)
        self.use_amp = self.device.type == "cuda" and cfg.LOFTR.MP

        self.model = LoFTR(config=lower_config(cfg)["loftr"]).to(self.device)
        state = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)["state_dict"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def prepare_image(self, image: Any) -> tuple[np.ndarray, torch.Tensor]:
        gray, tensor = _prepare_grayscale_image(image, self.long_side)
        return gray, tensor.to(self.device)

    def prepare_images(self, images: Sequence[Any]) -> list[tuple[np.ndarray, torch.Tensor]]:
        return [self.prepare_image(image) for image in images]

    def _forward_batch(self, image0_batch: torch.Tensor, image1_batch: torch.Tensor) -> dict[str, Any]:
        batch = {
            "image0": image0_batch,
            "image1": image1_batch,
        }

        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", enabled=self.use_amp):
                    self.model(batch)
            else:
                self.model(batch)

        return batch

    def _result_from_batch(
        self,
        batch: dict[str, Any],
        batch_index: int,
        image0: np.ndarray,
        image1: np.ndarray,
    ) -> MatchResult:
        match_mask = batch["m_bids"] == batch_index
        mkpts0 = batch["mkpts0_f"][match_mask].detach().cpu().numpy()
        mkpts1 = batch["mkpts1_f"][match_mask].detach().cpu().numpy()
        confidence = batch["mconf"][match_mask].detach().cpu().numpy()

        return MatchResult(
            mkpts0=mkpts0,
            mkpts1=mkpts1,
            confidence=confidence,
            covis0=_resize_covisibility_map(batch["matchability_score_list0"][-1][batch_index, 0], image0.shape),
            covis1=_resize_covisibility_map(batch["matchability_score_list1"][-1][batch_index, 0], image1.shape),
            image0=image0,
            image1=image1,
        )

    def match_batch(
        self,
        images: Sequence[Any],
        pair_indices: Sequence[tuple[int, int]],
    ) -> list[MatchResult]:
        if not pair_indices:
            return []

        if not images:
            raise ValueError("images must not be empty when pair_indices are provided")

        prepared_images = self.prepare_images(images)
        num_images = len(prepared_images)

        normalized_pairs: list[tuple[int, int]] = []
        for pair_index, (image0_index, image1_index) in enumerate(pair_indices):
            if not (0 <= image0_index < num_images):
                raise IndexError(
                    f"pair_indices[{pair_index}][0]={image0_index} is out of range for {num_images} images"
                )
            if not (0 <= image1_index < num_images):
                raise IndexError(
                    f"pair_indices[{pair_index}][1]={image1_index} is out of range for {num_images} images"
                )
            normalized_pairs.append((image0_index, image1_index))

        grouped_indices: dict[tuple[tuple[int, int], tuple[int, int]], list[int]] = {}
        for index, (image0_index, image1_index) in enumerate(normalized_pairs):
            tensor0 = prepared_images[image0_index][1]
            tensor1 = prepared_images[image1_index][1]
            shape_key = (
                (int(tensor0.shape[-2]), int(tensor0.shape[-1])),
                (int(tensor1.shape[-2]), int(tensor1.shape[-1])),
            )
            grouped_indices.setdefault(shape_key, []).append(index)

        results: list[MatchResult | None] = [None] * len(normalized_pairs)
        for indices in grouped_indices.values():
            image0_batch = torch.cat([prepared_images[normalized_pairs[index][0]][1] for index in indices], dim=0)
            image1_batch = torch.cat([prepared_images[normalized_pairs[index][1]][1] for index in indices], dim=0)
            batch = self._forward_batch(image0_batch, image1_batch)

            for local_index, original_index in enumerate(indices):
                image0_index, image1_index = normalized_pairs[original_index]
                image0 = prepared_images[image0_index][0]
                image1 = prepared_images[image1_index][0]
                results[original_index] = self._result_from_batch(batch, local_index, image0, image1)

        return [result for result in results if result is not None]

    def match_images(self, img_a: Any, img_b: Any) -> MatchResult:
        return self.match_batch([img_a, img_b], [(0, 1)])[0]


@lru_cache(maxsize=4)
def _cached_matcher(
    checkpoint_path: str,
    device: str | None,
    long_side: int,
    use_amp: bool | None,
    config_path: str | None,
    enable_flash: bool,
) -> CoMatchMatcher:
    return CoMatchMatcher(
        checkpoint_path,
        device=device,
        long_side=long_side,
        use_amp=use_amp,
        config_path=config_path,
        enable_flash=enable_flash,
    )


def match_batch(
    images: Sequence[Any],
    pair_indices: Sequence[tuple[int, int]],
    *,
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    long_side: int = 1152,
    use_amp: bool | None = None,
    config_path: str | Path | None = None,
    enable_flash: bool = True,
) -> list[MatchResult]:
    matcher = _cached_matcher(
        str(checkpoint_path),
        None if device is None else str(device),
        long_side,
        use_amp,
        None if config_path is None else str(config_path),
        enable_flash,
    )
    return matcher.match_batch(images, pair_indices)


def match_images(
    img_a: Any,
    img_b: Any,
    *,
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    long_side: int = 1152,
    use_amp: bool | None = None,
    config_path: str | Path | None = None,
    enable_flash: bool = True,
) -> MatchResult:
    matcher = _cached_matcher(
        str(checkpoint_path),
        None if device is None else str(device),
        long_side,
        use_amp,
        None if config_path is None else str(config_path),
        enable_flash,
    )
    return matcher.match_images(img_a, img_b)