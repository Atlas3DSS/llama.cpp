#pragma once

#include "common.cuh"

// Atlas Engine v7: Dedicated SM75+ Tensor-Core FlashAttention (DKQ=128, DV=128) with:
//   1. On-the-fly Inkling-2 relative-position bias (src[6]=rel_table, src[7]=k_pos, src[8]=q_pos)
//      eliminating the 19.2 GB kq_b buffer, per-head mask expansion, and 2D self_rel_pos PCIe copies,
//      while restoring 3x GQA KV sharing (mask->ne[2] == 1).
//   2. Direct in-shared-memory F8_E4M3 (double-buffered cp.async + branchless e4m3x8_h8) and
//      Q8_0 (F32 scale multiply + half round) KV tile loaders, eliminating need_f16_K/V HBM dequant.

bool ggml_cuda_flash_attn_ext_mma_f16_rel_supported(int device, const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_mma_f16_rel(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
