import pytest
import torch


def router_gemm_ref(input, weight, bias, dtype):
    logits_ref = torch.matmul(input, weight)
    return logits_ref


@pytest.mark.parametrize("num_tokens", [4])
@pytest.mark.parametrize("num_experts", [256])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_router_gemm_run(num_tokens, num_experts, hidden_size, dtype):
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)

    device = torch.device("cuda")
    input = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.randn((num_experts, hidden_size), dtype=dtype, device=device)
    bias = None
    logits = torch.ops.trtllm.router_gemm_op(input, weight.t(), bias,
                                             torch.float32)
    logtis_ref = router_gemm_ref(input, weight.t(), bias, dtype)
    assert torch.allclose(logits, logtis_ref)
