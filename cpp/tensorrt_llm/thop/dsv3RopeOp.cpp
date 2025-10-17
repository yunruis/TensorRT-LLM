#include "tensorrt_llm/common/attentionOp.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/mlaKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/debugUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

namespace tk = tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;
using tk::KVBlockArray;
using tk::KvCacheDataType;

namespace torch_ext
{

/* python op fake code

class MLA：
    ...
    def forward_generation(self):
        trtllm_attention.apply_rope_generation(
            fused_q,
            latent_cache,
            attn_metadata
        )

class trtllm_attention:
    def apply_rope_generation(self, fused_q, latent_cache, attn_metadata):
        fused_q = fused_q.view([-1, self.num_heads, self.kv_lora_rank + self.qk_rope_head_dim])
        q_pe = fused_q[..., -q_rope_dim:]
        kv_cache_lengths = metadata.kv_lens_cuda_runtime,  # [batch_size], 又叫 sequence_length

        batch_size = kv_cache_lengths.shape[0]  # 是否包含
        assert self.host_past_key_value_lengths.shape[0] == batch_size
        assert self.context_lengths.shape[0] == batch_size
        assert self.host_context_lengths.shape[0] == batch_size
        assert self.host_request_types.shape[0] == batch_size

        total_s_len = fused_q.shape[0]

        quant_scale_o = self.out_scale_sf if self.use_nvfp4_output else self.out_scale
        quant_scale_q = self.kv_scale_orig_quant
        quant_scale_kv = self.kv_scale_orig_quant
        dequant_scale_q = self.kv_scale_orig_quant
        dequant_scale_kv = self.kv_scale_quant_orig
        host_bmm1_scale = 1 / (self.q_scaling * sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim))

        # output
        workspace

        tensorrt_llm.op.MLARopeGeneration(
            # input
            q_pe, fused_buf, latent_cache, cos_sin_cache
            self.num_heads, self.kv_lora_rank,
            total_s_len,
            kv_cache_lengths,

            # kv_cache related
            self.quant_mode, self.kv_cache_block_offsets, self.host_kv_cache_block_offsets,
            self.host_kv_cache_pool_pointers,  self.host_kv_cache_pool_mapping,

            quant_scale_o, quant_scale_q, quant_scale_kv,
            dequant_scale_q, dequant_scale_kv,
            host_bmm1_scale,

            # output
            fused_q, quant_q, # kv_cache,
            workspace,
            # kv cache related
            self.beam_width
        )   # workspace 内部子缓冲：seqQOffset, cu_kv_seqlens, fmha_tile_counter, bmm1_scale, bmm2_scale,
quant_q_buf（FP8 MLA）。 # kv_cache_
*/

size_t getWorkspaceSizeForMLAGeneration(int const batch_size, bool const fp8_generation_mla, int const total_s_len,
    int const num_heads, int const kv_lora_rank, int const qk_rope_head_dim,
    uintptr_t const alignment = tc::kCudaMemAlign)
{
    size_t const cu_seqlens_size_aligned = tc::alignSize(sizeof(int) * (batch_size + 1), alignment);
    size_t const fmha_scheduler_counter_aligned = tc::alignSize(sizeof(uint32_t), alignment);
    size_t const mla_bmm1_scale_size_aligned = tc::alignSize(fp8_generation_mla ? sizeof(float) * 2 : 0, alignment);
    size_t const mla_bmm2_scale_size_aligned = tc::alignSize(fp8_generation_mla ? sizeof(float) : 0, alignment);
    size_t const quant_q_buffer_size_aligned = tc::alignSize(
        fp8_generation_mla ? total_s_len * size_t(num_heads * (kv_lora_rank + qk_rope_head_dim)) : 0, alignment);
    printf("alignment: %ld\n", alignment);
    printf("cu_seqlens_size_aligned: %ld\n", cu_seqlens_size_aligned);
    printf("fmha_scheduler_counter_aligned: %ld\n", fmha_scheduler_counter_aligned);
    printf("mla_bmm1_scale_size_aligned: %ld\n", mla_bmm1_scale_size_aligned);
    printf("mla_bmm2_scale_size_aligned: %ld\n", mla_bmm2_scale_size_aligned);
    printf("quant_q_buffer_size_aligned: %ld\n", quant_q_buffer_size_aligned);
    size_t const workspace_size = 2 * cu_seqlens_size_aligned + fmha_scheduler_counter_aligned
        + mla_bmm1_scale_size_aligned + mla_bmm2_scale_size_aligned + quant_q_buffer_size_aligned;
    return workspace_size;
}

// Wrapper for MLA rope generation arguments
struct MlaRopeGenArgs
{
    int32_t q_pe_ld;
    int32_t q_pe_stride;
    float2 const* rotary_cos_sin_ptr;
    int32_t num_seqs;
    int32_t num_tokens;
    int32_t num_heads;
    tk::MlaMetaParams mla_meta_params;
    int32_t const* sequence_lengths_ptr;
    int32_t max_context_q_len;
    int const* block_ids_per_seq_ptr;
    KvCacheDataType cache_type;
    int* cu_q_seqlens_ptr;
    int* cu_kv_seqlens_ptr;
    uint32_t* fmha_tile_counter_ptr;
    float* mla_bmm1_scale_ptr;
    float* mla_bmm2_scale_ptr;
    void* quant_q_buffer_ptr;
    float const* quant_scale_o_ptr;
    float const* kv_scale_orig_quant_ptr;
    float const* kv_scale_quant_orig_ptr;
    float host_bmm1_scale;
};

template <typename T, typename KVCacheBuffer>
void invokeMLARopeGenerationHelper(T const* latent_cache_ptr, T* q_pe_ptr, T* fused_q_ptr,
    KVCacheBuffer& kv_cache_buffer, MlaRopeGenArgs const& args, cudaStream_t stream)
{
    tk::MlaParams<T> mla_params{};
    mla_params.latent_cache = latent_cache_ptr;
    mla_params.q_pe = q_pe_ptr;
    mla_params.q_pe_ld = args.q_pe_ld;
    mla_params.q_pe_stride = args.q_pe_stride;
    mla_params.q_buf = fused_q_ptr;
    mla_params.cos_sin_cache = args.rotary_cos_sin_ptr;
    mla_params.batch_size = args.num_seqs;
    mla_params.acc_q_len = args.num_tokens;
    mla_params.head_num = args.num_heads;
    mla_params.meta = args.mla_meta_params;

    mla_params.cache_seq_lens = args.sequence_lengths_ptr;
    mla_params.max_input_seq_len = args.max_context_q_len;

    mla_params.block_ids_per_seq = args.block_ids_per_seq_ptr;

    // mlaGeneration()
    mla_params.cache_type = args.cache_type;

    mla_params.seqQOffset = args.cu_q_seqlens_ptr;
    mla_params.cu_kv_seqlens = args.cu_kv_seqlens_ptr;
    mla_params.fmha_tile_counter = args.fmha_tile_counter_ptr;
    mla_params.bmm1_scale = args.mla_bmm1_scale_ptr;
    mla_params.bmm2_scale = args.mla_bmm2_scale_ptr;
    mla_params.quant_q_buf = args.quant_q_buffer_ptr;

    mla_params.quant_scale_o = args.quant_scale_o_ptr;
    mla_params.quant_scale_q = args.kv_scale_orig_quant_ptr;
    mla_params.quant_scale_kv = args.kv_scale_orig_quant_ptr;
    mla_params.dequant_scale_q = args.kv_scale_quant_orig_ptr;
    mla_params.dequant_scale_kv = args.kv_scale_quant_orig_ptr;
    mla_params.host_bmm1_scale = args.host_bmm1_scale;

    tk::invokeMLARopeGeneration<T>(mla_params, kv_cache_buffer, stream);
}

/*
此处需要处理的：
input：
    q_pe_ld, q_pe_stride,
    cache_type
output:
    workspace:
        scale, ...
    fused_q: [q_len, 128 * 576] gen only
    kv_cache

验证coverage
    模型
        dsv3, dsv3_lite
    runtime：
        ctx only, gen only, continuous batchign
    gen_tokens
        1, mtp, others...
*/
void MLARopeGeneration(torch::Tensor fused_q, // [q_len, 128 * 576] gen only
    torch::Tensor q_pe,                       // [q_len, 128, 64] gen only
    torch::Tensor latent_cache,               // [q_len, 576] gen only
    std::optional<torch::Tensor> rotary_cos_sin, torch::Tensor cu_q_seqlens, torch::Tensor cu_kv_seqlens,
    torch::Tensor fmha_scheduler_counter, std::optional<torch::Tensor> mla_bmm1_scale,
    std::optional<torch::Tensor> mla_bmm2_scale, std::optional<torch::Tensor> quant_q_buffer,
    // kv cache related
    torch::Tensor sequence_length, torch::Tensor host_past_key_value_lengths, torch::Tensor host_context_lengths,

    std::optional<torch::Tensor> kv_cache_block_offsets, std::optional<torch::Tensor> host_kv_cache_block_offsets,
    std::optional<torch::Tensor> host_kv_cache_pool_pointers, std::optional<torch::Tensor> host_kv_cache_pool_mapping,
    // fp8 attn related
    torch::optional<torch::Tensor> kv_scale_orig_quant, // [1] q,k quant scale
    torch::optional<torch::Tensor> kv_scale_quant_orig, // [1] bmm quant scale
    torch::optional<torch::Tensor> out_scale,           // [1] bmm quant scale
    // rope related
    std::optional<torch::Tensor> block_ids_per_seq,

    int64_t const predicted_tokens_per_seq, int64_t const layer_idx, int64_t const num_heads,
    int64_t const num_kv_heads, // should be 1
    int64_t const head_size,

    int64_t const tokens_per_block, int64_t const attention_window_size, int64_t const sink_token_length,
    int64_t const beam_width, int64_t const quant_mode, double const q_scaling, int64_t q_lora_rank,
    int64_t kv_lora_rank, int64_t qk_nope_head_dim, int64_t qk_rope_head_dim, int64_t v_head_dim)
{
    auto stream = at::cuda::getCurrentCUDAStream(fused_q.get_device());
    /*
    前序校验工作
        1. 不允许nvfp4
        2. 必须启用kv cache
    */
    // 1. from thop::attentionOp.cpp::attention(): do prepare work，such as ctx, gen split, quant_type, workspace, slice
    // ctx, gen
    printf("=================MLARopeGeneration============");
    TLLM_CHECK_WITH_INFO(
        head_size == kv_lora_rank + qk_rope_head_dim, "head_size must = kv_lora_rank + qk_rope_head_dim");
    TLLM_CHECK_WITH_INFO(num_kv_heads == 1, "num_kv_heads must = 1");

    auto const kv_cache_quant_mode = tc::QuantMode(uint32_t(quant_mode));
    bool const fp8_generation_mla = kv_cache_quant_mode.hasFp8KvCache();
    bool const use_gen_flash_mla = tc::getSMVersion() == 90 && tokens_per_block == 64;
    TLLM_CHECK_WITH_INFO(!kv_cache_quant_mode.hasFp4KvCache(), "FP4 KV cache is not supported for MLA generation.");
    TLLM_CHECK_WITH_INFO(
        host_kv_cache_pool_mapping.has_value(), "KV cache pool mapping is required for MLA generation.");

    bool const use_kv_cache = kv_cache_block_offsets.has_value() && host_kv_cache_block_offsets.has_value()
        && host_kv_cache_pool_pointers.has_value() && host_kv_cache_pool_mapping.has_value();

    // continuous batching related

    int32_t const num_seqs = host_context_lengths.size(0); // num_seqs = num_ctx + num_gen
    int32_t const num_contexts = 0;
    int32_t const num_generations = num_seqs - num_contexts;
    int32_t const num_gen_tokens = fused_q.size(0);
    int32_t const num_tokens = num_gen_tokens;
    int32_t const seq_offset = num_contexts;

    // model config related
    int32_t const layer_num = host_kv_cache_pool_mapping.value().size(0);

    tk::MlaMetaParams mla_meta_params = {static_cast<int>(q_lora_rank), static_cast<int>(kv_lora_rank),
        static_cast<int>(qk_nope_head_dim), static_cast<int>(qk_rope_head_dim), static_cast<int>(v_head_dim),
        static_cast<int>(predicted_tokens_per_seq), static_cast<int>(layer_num)};

    int* cu_q_seqlens_ptr = reinterpret_cast<int*>(cu_q_seqlens.data_ptr());
    int* cu_kv_seqlens_ptr = reinterpret_cast<int*>(cu_kv_seqlens.data_ptr());
    uint32_t* fmha_tile_counter_ptr = reinterpret_cast<uint32_t*>(fmha_scheduler_counter.data_ptr());
    float* mla_bmm1_scale_ptr
        = mla_bmm1_scale.has_value() ? reinterpret_cast<float*>(mla_bmm1_scale.value().data_ptr()) : nullptr;
    float* mla_bmm2_scale_ptr
        = mla_bmm2_scale.has_value() ? reinterpret_cast<float*>(mla_bmm2_scale.value().data_ptr()) : nullptr;
    void* quant_q_buffer_ptr
        = quant_q_buffer.has_value() ? reinterpret_cast<void*>(quant_q_buffer.value().data_ptr()) : nullptr;

    // pointer preparation, stride,
    float2 const* rotary_cos_sin_ptr = nullptr;
    if (rotary_cos_sin.has_value())
    {
        rotary_cos_sin_ptr = reinterpret_cast<float2 const*>(rotary_cos_sin.value().data_ptr());
    }

    int const* sequence_lengths_ptr = sequence_length.slice(0, seq_offset).data_ptr<int>();
    // Note we still need context length during generation for MMHA optimization.
    int32_t const max_context_q_len
        = host_context_lengths.slice(0, seq_offset, seq_offset + num_seqs).max().item<int32_t>();

    TORCH_CHECK(q_pe.defined());
    TORCH_CHECK(q_pe.dim() == 3);
    TORCH_CHECK(q_pe.strides()[2] == 1);
    int32_t const q_pe_ld = q_pe.strides()[1];
    int32_t const q_pe_stride = q_pe.strides()[0];

    // kv cache related
    auto const block_size = tokens_per_block * num_kv_heads * head_size;
    int32_t const elem_bytes
        = kv_cache_quant_mode.hasFp8KvCache() ? sizeof(__nv_fp8_e4m3) : static_cast<int32_t>(fused_q.element_size());

    int32_t const bytes_per_token = num_kv_heads * head_size * elem_bytes;

    auto const bytes_per_block = block_size * elem_bytes;
    int32_t const kv_factor = 1; // 1 for mla, 2 for mha/gqa
    bool const fp8_context_fmha = kv_cache_quant_mode.hasFp8KvCache();
    // Commonly, cyclic_attention_window_size, and max_attention_window_size will be the same
    // unless each layer has different attention window sizes.
    // the kv_cache capacity.
    // The cyclic_attention_window_size will determine the cyclic kv cache position of new tokens.
    // Note that this cyclic_attention_window_size might be smaller than the actual kv cache capactity.
    int const cyclic_attention_window_size = attention_window_size;
    int const max_cyclic_attention_window_size = cyclic_attention_window_size;
    bool const can_use_one_more_block = beam_width > 1;

    // kv cache pool related
    int32_t const max_blocks_per_sequence
        = use_kv_cache && kv_cache_block_offsets.has_value() ? kv_cache_block_offsets.value().size(-1) : 0;
    int32_t const pool_index = use_kv_cache && host_kv_cache_pool_mapping.has_value()
        ? host_kv_cache_pool_mapping.value().index({layer_idx, 0}).item<int32_t>()
        : 0;
    int32_t const layer_idx_in_cache_pool = use_kv_cache && host_kv_cache_pool_mapping.has_value()
        ? host_kv_cache_pool_mapping.value().index({layer_idx, 1}).item<int32_t>()
        : 0;
    auto const intra_pool_offset = layer_idx_in_cache_pool * kv_factor * bytes_per_block;

    KVBlockArray::DataType* block_offsets
        = static_cast<KVBlockArray::DataType*>(use_kv_cache && kv_cache_block_offsets.has_value()
                ? kv_cache_block_offsets.value().index({pool_index, seq_offset}).data_ptr()
                : nullptr);

    // Prepare block pool pointers for NVFP4 KV cache.
    void* host_primary_pool_pointer{nullptr};
    void* host_secondary_pool_pointer{nullptr};

    if (use_kv_cache)
    {
        TORCH_CHECK(host_kv_cache_pool_pointers.value().dim() == 2);
        host_primary_pool_pointer = reinterpret_cast<void*>(
            reinterpret_cast<char*>(host_kv_cache_pool_pointers.value().index({pool_index, 0}).item<int64_t>())
            + intra_pool_offset);
        host_secondary_pool_pointer = reinterpret_cast<void*>(
            reinterpret_cast<char*>(host_kv_cache_pool_pointers.value().index({pool_index, 1}).item<int64_t>())
            + intra_pool_offset);
    }

    float const* kv_scale_orig_quant_ptr = nullptr; // qk quant scale
    float const* kv_scale_quant_orig_ptr = nullptr; // bmm quant scale
    if (kv_cache_quant_mode.hasKvCacheQuant() && kv_scale_orig_quant.has_value() && kv_scale_quant_orig.has_value())
    {
        kv_scale_orig_quant_ptr = kv_scale_orig_quant.value().data_ptr<float>();
        kv_scale_quant_orig_ptr = kv_scale_quant_orig.value().data_ptr<float>();
    }

    // Prepare scalars for MLA params and wrapper
    // For FP8 output, out_scale represents the output scale.
    float const* quant_scale_o_ptr
        = (fp8_context_fmha && out_scale.has_value()) ? out_scale.value().data_ptr<float>() : nullptr;
    float const host_bmm1_scale = 1.f / (q_scaling * sqrt(static_cast<float>(qk_nope_head_dim + qk_rope_head_dim)));

    if (use_gen_flash_mla)
    {
        TLLM_CHECK_WITH_INFO(block_ids_per_seq.has_value(), "block_ids_per_seq is required for gen flash mla");
    }
    int const* block_ids_per_seq_ptr = use_gen_flash_mla && block_ids_per_seq.has_value()
        ? static_cast<int*>(block_ids_per_seq->data_ptr())
        : nullptr; // only used for flash mla

    // attention sink, not used

    // 3. mla_generation()
    int32_t const batch_beam = beam_width * num_generations;

    KvCacheDataType cache_type = (kv_cache_quant_mode.hasFp8KvCache() ? KvCacheDataType::FP8 : KvCacheDataType::BASE);

    auto kv_cache_buffer = KVBlockArray(batch_beam, max_blocks_per_sequence, tokens_per_block, bytes_per_token,
        cyclic_attention_window_size, max_cyclic_attention_window_size, sink_token_length, can_use_one_more_block,
        host_primary_pool_pointer, host_secondary_pool_pointer, block_offsets);
    // Currently NVFP4 KV cache is not supported for MLA. An empty placeholder is provided.

    MlaRopeGenArgs args{q_pe_ld, q_pe_stride, rotary_cos_sin_ptr, num_seqs, num_tokens, static_cast<int32_t>(num_heads),
        mla_meta_params, sequence_lengths_ptr, max_context_q_len, block_ids_per_seq_ptr, cache_type, cu_q_seqlens_ptr,
        cu_kv_seqlens_ptr, fmha_tile_counter_ptr, mla_bmm1_scale_ptr, mla_bmm2_scale_ptr, quant_q_buffer_ptr,
        quant_scale_o_ptr, kv_scale_orig_quant_ptr, kv_scale_quant_orig_ptr, host_bmm1_scale};

    auto const input_dtype = fused_q.scalar_type();
    if (input_dtype == torch::kFloat16)
    {
        invokeMLARopeGenerationHelper(static_cast<half const*>(latent_cache.data_ptr()),
            static_cast<half*>(q_pe.data_ptr()), static_cast<half*>(fused_q.data_ptr()), kv_cache_buffer, args, stream);
    }
    else if (input_dtype == torch::kBFloat16)
    {

        invokeMLARopeGenerationHelper(static_cast<__nv_bfloat16 const*>(latent_cache.data_ptr()),
            static_cast<__nv_bfloat16*>(q_pe.data_ptr()), static_cast<__nv_bfloat16*>(fused_q.data_ptr()),
            kv_cache_buffer, args, stream);
    }
    else if (input_dtype == torch::kFloat32)
    {
        invokeMLARopeGenerationHelper(static_cast<float const*>(latent_cache.data_ptr()),
            static_cast<float*>(q_pe.data_ptr()), static_cast<float*>(fused_q.data_ptr()), kv_cache_buffer, args,
            stream);
    }
    else
    {
        TLLM_LOG_ERROR("Unsupported input dtype: %s", c10::toString(input_dtype));
    }
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mla_rope_generation("
        "Tensor(a!) fused_q"
        ", Tensor(a!) q_pe"
        ", Tensor latent_cache"
        ", Tensor? rotary_cos_sin"
        ", Tensor cu_q_seqlens"
        ", Tensor cu_kv_seqlens"
        ", Tensor fmha_scheduler_counter"
        ", Tensor? mla_bmm1_scale"
        ", Tensor? mla_bmm2_scale"
        ", Tensor? quant_q_buffer"
        ", Tensor sequence_length"
        ", Tensor host_past_key_value_lengths"
        ", Tensor host_context_lengths"
        ", Tensor? kv_cache_block_offsets"
        ", Tensor? host_kv_cache_block_offsets"
        ", Tensor? host_kv_cache_pool_pointers"
        ", Tensor? host_kv_cache_pool_mapping"
        ", Tensor? kv_scale_orig_quant"
        ", Tensor? kv_scale_quant_orig"
        ", Tensor? out_scale"
        ", Tensor? block_ids_per_seq"
        ", int predicted_tokens_per_seq"
        ", int layer_idx"
        ", int num_heads"
        ", int num_kv_heads"
        ", int head_size"
        ", int tokens_per_block"
        ", int attention_window_size"
        ", int sink_token_length"
        ", int beam_width"
        ", int quant_mode"
        ", float q_scaling"
        ", int q_lora_rank"
        ", int kv_lora_rank"
        ", int qk_nope_head_dim"
        ", int qk_rope_head_dim"
        ", int v_head_dim"
        ") -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mla_rope_generation", &torch_ext::MLARopeGeneration);
}
