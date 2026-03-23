import torch
import torch.nn as nn

from ..utils.misc import detect_NaN


class FixedSizeLoFTRExport(nn.Module):
    def __init__(self, matcher, height, width):
        super().__init__()
        self.matcher = matcher
        self.height = height
        self.width = width

    def _validate_input(self, image, name):
        expected_shape = (1, 1, self.height, self.width)
        if tuple(image.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(image.shape)}")

    def forward(self, image0, image1):
        self._validate_input(image0, "image0")
        self._validate_input(image1, "image1")

        data = {
            "image0": image0,
            "image1": image1,
            "bs": 1,
            "hw0_i": image0.shape[2:],
            "hw1_i": image1.shape[2:],
        }

        ret_dict = self.matcher.backbone(torch.cat([image0, image1], dim=0))
        feats_c = ret_dict["feats_c"]
        data.update({
            "feats_x2": ret_dict["feats_x2"],
            "feats_x1": ret_dict["feats_x1"],
        })
        feat_c0, feat_c1 = feats_c.split(1)

        mul = self.matcher.config["resolution"][0] // self.matcher.config["resolution"][1]
        data.update({
            "hw0_c": feat_c0.shape[2:],
            "hw1_c": feat_c1.shape[2:],
            "hw0_f": [feat_c0.shape[2] * mul, feat_c0.shape[3] * mul],
            "hw1_f": [feat_c1.shape[2] * mul, feat_c1.shape[3] * mul],
        })

        feat_c0, feat_c1, matchability_score_list0, matchability_score_list1 = self.matcher.loftr_coarse(
            feat_c0,
            feat_c1,
        )
        data.update({
            "matchability_score_list0": matchability_score_list0,
            "matchability_score_list1": matchability_score_list1,
        })

        feat_c0 = feat_c0.flatten(2).transpose(1, 2)
        feat_c1 = feat_c1.flatten(2).transpose(1, 2)

        if self.matcher.config["replace_nan"] and (
            torch.any(torch.isnan(feat_c0)) or torch.any(torch.isnan(feat_c1))
        ):
            detect_NaN(feat_c0, feat_c1)

        self.matcher.coarse_matching(feat_c0, feat_c1, data)

        feat_c0 = feat_c0 / feat_c0.shape[-1] ** 0.5
        feat_c1 = feat_c1 / feat_c1.shape[-1] ** 0.5

        feat_f0_unfold, feat_f1_unfold = self.matcher.fine_preprocess(feat_c0, feat_c1, data)

        if self.matcher.config["replace_nan"] and (
            torch.any(torch.isnan(feat_f0_unfold)) or torch.any(torch.isnan(feat_f1_unfold))
        ):
            detect_NaN(feat_f0_unfold, feat_f1_unfold)

        if feat_f0_unfold.size(0) != 0:
            feat_f0_unfold, feat_f1_unfold = self.matcher.loftr_fine(feat_f0_unfold, feat_f1_unfold)

        self.matcher.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

        return (
            data["mkpts0_f"],
            data["mkpts1_f"],
            data["mconf"],
            data["matchability_score_list0"][-1],
            data["matchability_score_list1"][-1],
        )