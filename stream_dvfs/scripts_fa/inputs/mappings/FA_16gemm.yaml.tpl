- name: default
  core_allocation: [0]

- name: MatMul
  core_allocation: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  intra_core_tiling:
    - D, <num_qkv_tiles>
  inter_core_tiling:
    - B, 1
- name: FA_Gemm
  core_allocation: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  intra_core_tiling:
    - BATCH, 1
  inter_core_tiling:
    - BR, 1
- name: FA_Simd
  core_allocation: [16]
  intra_core_tiling:
    - BATCH, 1
  inter_core_tiling:
    - BR, 1
