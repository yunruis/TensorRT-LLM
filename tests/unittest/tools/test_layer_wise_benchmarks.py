import os

import pytest
import torch
from defs.conftest import deepseek_r1_model_root  # noqa: F401
from defs.conftest import deepseek_v3_model_root  # noqa: F401
from defs.trt_test_alternative import check_call
from utils.cpp_paths import llm_root  # noqa: F401


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="needs 4 GPUs to run this test")
@pytest.mark.parametrize("deepseek_r1_model_root", ["DeepSeek-R1-0528-FP4-v2"],
                         indirect=True)
def test_deepseek_r1_ctx_tep(llm_root, deepseek_r1_model_root):
    check_call([
        "./mpi_launch.sh",
        "./run_single.sh",
        "config_ctx.yaml",
        "--model=" + deepseek_r1_model_root,
        "--no-enable-attention-dp",
        "--moe-backend=TRTLLM",
    ],
               cwd=llm_root / "examples" / "layer_wise_benchmarks",
               env={
                   **os.environ,
                   "NP": "4",
                   "TRTLLM_ENABLE_PDL": "1",
               })


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="needs 4 GPUs to run this test")
@pytest.mark.parametrize("deepseek_v3_model_root", ["DeepSeek-V3.2-Exp"],
                         indirect=True)
def test_deepseek_v32_ctx_dep(llm_root, deepseek_v3_model_root):
    check_call([
        "./mpi_launch.sh",
        "./run_single.sh",
        "config_ctx.yaml",
        "--model=" + deepseek_v3_model_root,
        "--tokens-per-block=64",
        "--moe-backend=DEEPGEMM",
    ],
               cwd=llm_root / "examples" / "layer_wise_benchmarks",
               env={
                   **os.environ,
                   "NP": "4",
               })


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="needs 4 GPUs to run this test")
@pytest.mark.parametrize("deepseek_r1_model_root", ["DeepSeek-R1-0528-FP4-v2"],
                         indirect=True)
def test_deepseek_r1_gen_scaled_from_16_dep(llm_root, deepseek_r1_model_root):
    check_call([
        "./mpi_launch.sh",
        "./run_single.sh",
        "config_gen.yaml",
        "--model=" + deepseek_r1_model_root,
        "--layer-indices=5,6",
        "--scaled-from=16",
        "--moe-backend=WIDEEP",
    ],
               cwd=llm_root / "examples" / "layer_wise_benchmarks",
               env={
                   **os.environ,
                   "NP": "4",
               })
