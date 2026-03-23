# ONNX / TensorRT Todo

## Low Hanging Fruit

- [x] Benchmark fair inference path
  - Scope: time only the real matcher forward, enable warmup, and compare AMP vs FP32.
  - Goal: get an apples-to-apples latency number before optimization work.
  - Status: implemented in `test_inference.py` with warmup iterations, timed iterations, and AMP toggle.

- [x] Replace Kornia DSNT
  - File: `src/loftr/utils/fine_matching_epipolar.py`
  - Scope: replace `dsnt.spatial_expectation2d` with a pure torch soft-argmax / expectation.
  - Status: replaced with a torch-native `_spatial_expectation2d` helper.

- [x] Replace Kornia meshgrid ops
  - Files:
    - `src/loftr/utils/fine_matching_epipolar.py`
    - `src/loftr/utils/geometry.py`
  - Scope: replace `create_meshgrid` with torch-native coordinate construction or fixed buffers.
  - Status: replaced in fine matching with a torch-native `_create_meshgrid` helper.

- [x] Replace geometry Kornia helper
  - File: `src/loftr/utils/geometry.py`
  - Scope: replace `numeric.cross_product_matrix` with a torch skew-matrix helper.
  - Status: replaced with a torch-native `_cross_product_matrix` helper.

- [x] Remove inference einops
  - Files:
    - `src/loftr/loftr.py`
    - `src/loftr/loftr_module/fine_preprocess_epipolar.py`
    - `src/loftr/loftr_module/transformer.py`
    - `src/loftr/loftr_module/linear_attention.py`
  - Scope: replace `rearrange` / `repeat` with `permute`, `reshape`, `flatten`, `expand`, and `repeat_interleave`.
  - Status: completed for the low-risk inference-path shape transforms.

- [x] Freeze fixed-size inference path
  - Scope: commit to batch size 1 and a fixed input size for export.
  - Goal: simplify ONNX/TensorRT graph generation.
  - Status: added `FixedSizeLoFTRExport` in `src/loftr/fixed_size_export.py` to expose a batch-1, fixed-resolution export path.

## Medium Difficulty

- [ ] Export ONNX smoke test
  - Scope: export the inference path after the low-hanging replacements and identify the first real graph blockers.

- [ ] Rewrite fine-stage indexing
  - File: `src/loftr/utils/fine_matching_epipolar.py`
  - Scope: replace advanced indexing with export-friendlier gather-based logic where possible.

- [ ] Rewrite preprocess match gathering
  - File: `src/loftr/loftr_module/fine_preprocess_epipolar.py`
  - Scope: make local-window extraction and match selection friendlier to ONNX export.

## Hard / Last

- [ ] Replace flash attention branch
  - File: `src/loftr/loftr_module/linear_attention.py`
  - Scope: replace the flash / SDP attention branch with a plain torch attention path for export stability.

- [ ] Vectorize attention masking path
  - File: `src/loftr/loftr_module/linear_attention.py`
  - Scope: remove batch loops and simplify crop/pad masking behavior for deployment.

- [ ] Build TensorRT prototype
  - Scope: only after ONNX export is stable.
  - Goal: validate whether the cleaned graph gives a meaningful latency win.

## Recommended Order

1. Benchmark fair inference path
2. Replace Kornia ops
3. Remove inference einops
4. Freeze fixed-size inference path
5. Export ONNX smoke test
6. Rewrite fine-stage indexing
7. Rewrite preprocess match gathering
8. Replace flash attention branch
9. Vectorize attention masking path
10. Build TensorRT prototype