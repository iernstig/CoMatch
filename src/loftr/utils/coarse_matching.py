import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

INF = 1e9


def _sync_if_needed(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _time_block(device, fn):
    _sync_if_needed(device)
    start_time = time.perf_counter()
    result = fn()
    _sync_if_needed(device)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return result, elapsed_ms

def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch
    
    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand

class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # general config
        self.thr = config['thr']
        self.border_rm = config['border_rm']
        
        
        self.temperature=nn.parameter.Parameter(torch.tensor(0.1), requires_grad=True)
        self.skip_softmax = config['skip_softmax']
        self.fp16matmul = config['fp16matmul']
        # -- # for trainig fine-level LoFTR
        self.train_coarse_percent = config['train_coarse_percent']
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']

    def _apply_dual_softmax(self, sim_matrix, softmax_mode):
        if softmax_mode == 'log_softmax':
            return torch.exp(F.log_softmax(sim_matrix, dim=1) + F.log_softmax(sim_matrix, dim=2))
        if softmax_mode == 'softmax':
            return F.softmax(sim_matrix, dim=1) * F.softmax(sim_matrix, dim=2)
        raise ValueError(f"Unsupported coarse softmax mode: {softmax_mode}")

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)
        profile_enabled = data.get('_profile_coarse_matching', False)
        softmax_mode = data.get('_coarse_softmax_mode', 'softmax')
        timings = {}
        device = feat_c0.device

        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5,
                               [feat_c0, feat_c1])

        if self.fp16matmul:
            if profile_enabled:
                sim_matrix, timings['similarity'] = _time_block(
                    device,
                    lambda: torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature,
                )
            else:
                sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0,
                                        feat_c1) / self.temperature
            del feat_c0, feat_c1
            if mask_c0 is not None:
                if profile_enabled:
                    sim_matrix, timings['mask_fill'] = _time_block(
                        device,
                        lambda: sim_matrix.masked_fill(
                            ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                            -1e4,
                        ),
                    )
                else:
                    sim_matrix = sim_matrix.masked_fill(
                        ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                        -1e4
                        )
        else:
            autocast_context = torch.autocast(enabled=False, device_type='cuda') if device.type == 'cuda' else nullcontext()
            with autocast_context:
                if profile_enabled:
                    sim_matrix, timings['similarity'] = _time_block(
                        device,
                        lambda: torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature,
                    )
                else:
                    sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0,
                                            feat_c1) / self.temperature
                del feat_c0, feat_c1
                if mask_c0 is not None:
                    if profile_enabled:
                        sim_matrix, timings['mask_fill'] = _time_block(
                            device,
                            lambda: sim_matrix.float().masked_fill(
                                ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                                -INF,
                            ),
                        )
                    else:
                        sim_matrix = sim_matrix.float().masked_fill(
                            ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                            -INF
                            )
        if self.skip_softmax:
            sim_matrix = sim_matrix
        else:
            if profile_enabled:
                sim_matrix, timings['softmax'] = _time_block(
                    device,
                    lambda: self._apply_dual_softmax(sim_matrix, softmax_mode),
                )
            else:
                sim_matrix = self._apply_dual_softmax(sim_matrix, softmax_mode)
       
        
        data.update({'conf_matrix': sim_matrix})

        # predict coarse matches from conf_matrix
        if profile_enabled:
            data['coarse_matching_profile_ms'] = timings
            coarse_matches, get_match_ms = _time_block(device, lambda: self.get_coarse_match(sim_matrix, data))
            timings['extract_matches'] = get_match_ms
            data['coarse_matching_profile_ms'] = timings
            data.update(**coarse_matches)
        else:
            data.update(**self.get_coarse_match(sim_matrix, data))

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix, data):
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }
        _device = conf_matrix.device
        profile_enabled = data.get('_profile_coarse_matching', False)
        timings = data.get('coarse_matching_profile_ms') if profile_enabled else None
        # 1. confidence thresholding
        if profile_enabled:
            mask, timings['threshold'] = _time_block(device=_device, fn=lambda: conf_matrix > self.thr)
            mask, timings['reshape_5d'] = _time_block(
                device=_device,
                fn=lambda: mask.reshape(
                    mask.shape[0],
                    axes_lengths['h0c'],
                    axes_lengths['w0c'],
                    axes_lengths['h1c'],
                    axes_lengths['w1c'],
                ),
            )
        else:
            mask = conf_matrix > self.thr
            mask = mask.reshape(
                mask.shape[0],
                axes_lengths['h0c'],
                axes_lengths['w0c'],
                axes_lengths['h1c'],
                axes_lengths['w1c'],
            )

        if 'mask0' not in data:
            if profile_enabled:
                _, timings['border_mask'] = _time_block(_device, lambda: mask_border(mask, self.border_rm, False))
            else:
                mask_border(mask, self.border_rm, False)
        else:
            if profile_enabled:
                _, timings['border_mask'] = _time_block(
                    _device,
                    lambda: mask_border_with_padding(mask, self.border_rm, False, data['mask0'], data['mask1']),
                )
            else:
                mask_border_with_padding(mask, self.border_rm, False,
                                         data['mask0'], data['mask1'])
        if profile_enabled:
            mask, timings['reshape_3d'] = _time_block(
                _device,
                lambda: mask.reshape(
                    mask.shape[0],
                    axes_lengths['h0c'] * axes_lengths['w0c'],
                    axes_lengths['h1c'] * axes_lengths['w1c'],
                ),
            )
        else:
            mask = mask.reshape(
                mask.shape[0],
                axes_lengths['h0c'] * axes_lengths['w0c'],
                axes_lengths['h1c'] * axes_lengths['w1c'],
            )
            
        # 2. mutual nearest
        if profile_enabled:
            row_max, timings['row_max'] = _time_block(_device, lambda: conf_matrix.max(dim=2, keepdim=True)[0])
            col_max, timings['col_max'] = _time_block(_device, lambda: conf_matrix.max(dim=1, keepdim=True)[0])
            mask, timings['mutual_nearest'] = _time_block(
                _device,
                lambda: mask * (conf_matrix == row_max) * (conf_matrix == col_max),
            )
        else:
            mask = mask \
                * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
                * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

        # 3. find all valid coarse matches
        # this only works when at most one `True` in each row
        if profile_enabled:
            (mask_v, all_j_ids), timings['mask_reduce'] = _time_block(_device, lambda: mask.max(dim=2))
            (b_ids, i_ids), timings['where'] = _time_block(_device, lambda: torch.where(mask_v))
            j_ids, timings['gather_j_ids'] = _time_block(_device, lambda: all_j_ids[b_ids, i_ids])
            mconf, timings['gather_conf'] = _time_block(_device, lambda: conf_matrix[b_ids, i_ids, j_ids])
        else:
            mask_v, all_j_ids = mask.max(dim=2)
            b_ids, i_ids = torch.where(mask_v)
            j_ids = all_j_ids[b_ids, i_ids]
            mconf = conf_matrix[b_ids, i_ids, j_ids]

        # 4. Random sampling of training samples for fine-level LoFTR
        # (optional) pad samples with gt coarse-level matches
        if self.training:
            # NOTE:
            # The sampling is performed across all pairs in a batch without manually balancing
            # #samples for fine-level increases w.r.t. batch_size
            if 'mask0' not in data:
                num_candidates_max = mask.size(0) * max(
                    mask.size(1), mask.size(2))
            else:
                num_candidates_max = compute_max_candidates(
                    data['mask0'], data['mask1'])
            num_matches_train = int(num_candidates_max *
                                    self.train_coarse_percent)
            num_matches_pred = len(b_ids)
            assert self.train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"

            # pred_indices is to select from prediction
            if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred,
                    (num_matches_train - self.train_pad_num_gt_min, ),
                    device=_device)

            # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
            gt_pad_indices = torch.randint(
                    len(data['spv_b_ids']),
                    (max(num_matches_train - num_matches_pred,
                        self.train_pad_num_gt_min), ),
                    device=_device)
            mconf_gt = torch.zeros(len(data['spv_b_ids']), device=_device)  # set conf of gt paddings to all zero

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]],
                                       dim=0),
                *zip([b_ids, data['spv_b_ids']], [i_ids, data['spv_i_ids']],
                     [j_ids, data['spv_j_ids']], [mconf, mconf_gt]))

        # These matches select patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # 4. Update with matches in original image resolution
        scale = data['hw0_i'][0] / data['hw0_c'][0]

        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        if profile_enabled:
            mkpts0_c, timings['decode_mkpts0'] = _time_block(
                _device,
                lambda: torch.stack(
                    [i_ids % data['hw0_c'][1], i_ids // data['hw0_c'][1]],
                    dim=1) * scale0,
            )
            mkpts1_c, timings['decode_mkpts1'] = _time_block(
                _device,
                lambda: torch.stack(
                    [j_ids % data['hw1_c'][1], j_ids // data['hw1_c'][1]],
                    dim=1) * scale1,
            )
        else:
            mkpts0_c = torch.stack(
                [i_ids % data['hw0_c'][1], i_ids // data['hw0_c'][1]],
                dim=1) * scale0 
            mkpts1_c = torch.stack(
                [j_ids % data['hw1_c'][1], j_ids // data['hw1_c'][1]],
                dim=1) * scale1 

        m_bids = b_ids[mconf != 0]        
        # These matches is the current prediction (for visualization)

        coarse_matches.update({
            'm_bids': m_bids,  # mconf == 0 => gt matches
            'all_mkpts0_c': mkpts0_c,
            'all_mkpts1_c': mkpts1_c,
            'mkpts0_c': mkpts0_c[mconf != 0],
            'mkpts1_c': mkpts1_c[mconf != 0],
            'mconf': mconf[mconf != 0]
        })
        

        return coarse_matches