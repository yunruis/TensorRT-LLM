#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Microbenchmark for torch.ops.trtllm.fused_inv_rope_fp8_quant_vllm_port.
#
# The timed region calls only this Torch op. Inputs are allocated once per
# problem size; warmup and benchmark loops repeatedly invoke the op and measure
# CUDA event elapsed time on the current stream. Optional CUDA Graph mode
# captures repeated op calls and reports per-op replay time.

import argparse
import statistics
from dataclasses import dataclass

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401  Registers trtllm Torch ops.
from tensorrt_llm._utils import is_sm_100f

"""
 python3 tests/microbenchmarks/bench_fused_inv_rope_fp8_quant.py \
    --tokens 1024,4096,8192,16384,32768 \
    --n-groups 16 --heads-per-group 8 \
    --nope-dim 448 --rope-dim 64 \
    --position-dtype int64 \
    --warmup-steps 50 --bench-steps 200 --repeats 3 \
    --use-cuda-graph --graph-capture-steps 16
"""


@dataclass(frozen=True)
class Problem:
    num_tokens: int
    n_groups: int
    heads_per_group: int
    nope_dim: int
    rope_dim: int
    is_neox: bool

    @property
    def num_heads(self) -> int:
        return self.n_groups * self.heads_per_group

    @property
    def head_dim(self) -> int:
        return self.nope_dim + self.rope_dim

    @property
    def chunks_per_head(self) -> int:
        return self.head_dim // 128


def parse_int_list(value: str) -> list[int]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark torch.ops.trtllm.fused_inv_rope_fp8_quant_vllm_port only."
    )
    parser.add_argument(
        "--tokens",
        type=parse_int_list,
        default=parse_int_list("1024,4096,8192,16384,32768"),
        help="Comma-separated num_tokens list. Default: PR #14547 benchmark sizes.",
    )
    parser.add_argument("--n-groups", type=int, default=8, help="Default: DSv4-Flash DEP8.")
    parser.add_argument("--heads-per-group", type=int, default=8, help="Default: DSv4 production.")
    parser.add_argument("--nope-dim", type=int, default=448, help="Default: DSv4 head_dim 512.")
    parser.add_argument(
        "--rope-dim", type=int, default=64, help="Only 64 is supported by the CUDA op."
    )
    parser.add_argument(
        "--is-neox",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="RoPE layout. Default true matches PR #14547; use --no-is-neox for GPT-J/interleaved.",
    )
    parser.add_argument("--quant-group-size", type=int, default=128, help="Only 128 is supported.")
    parser.add_argument(
        "--position-dtype",
        choices=("int64", "int32"),
        default="int64",
        help="Default int64 avoids the wrapper's int32->int64 upcast during timing.",
    )
    parser.add_argument(
        "--max-position", type=int, default=0, help="0 means max(2048, max(tokens)+16)."
    )
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--bench-steps", type=int, default=200)
    parser.add_argument(
        "--repeats", type=int, default=1, help="Repeat timed loop and report median."
    )
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        help="Capture the op in a CUDA graph and time graph replay.",
    )
    parser.add_argument(
        "--graph-capture-steps",
        type=int,
        default=16,
        help="Number of op calls captured in one graph. Per-op time is elapsed / "
        "(bench_steps * graph_capture_steps). Only used with --use-cuda-graph.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--allow-non-sm100",
        action="store_true",
        help="Skip the SM100 guard. The production call site is SM100-gated.",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Call cudaProfilerStart/Stop around the timed region. Use with "
        "nsys profile --capture-range=cudaProfilerApi.",
    )
    return parser.parse_args()


def make_inputs(
    problem: Problem, args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda", args.device)
    max_position = args.max_position or max(2048, max(args.tokens) + 16)
    half_rope = problem.rope_dim // 2

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + problem.num_tokens)

    o = torch.randn(
        problem.num_tokens,
        problem.num_heads,
        problem.head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    ).contiguous()

    position_dtype = torch.int64 if args.position_dtype == "int64" else torch.int32
    positions = torch.randint(
        0,
        max_position,
        (problem.num_tokens,),
        device=device,
        dtype=position_dtype,
        generator=gen,
    )

    theta = torch.randn(max_position, half_rope, device=device, dtype=torch.float32, generator=gen)
    rotary_cos_sin = torch.stack([theta.cos(), theta.sin()], dim=1).contiguous()
    return o, positions, rotary_cos_sin


def call_fused(
    o: torch.Tensor,
    positions: torch.Tensor,
    rotary_cos_sin: torch.Tensor,
    problem: Problem,
    quant_group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.trtllm.fused_inv_rope_fp8_quant_vllm_port(
        o,
        positions,
        rotary_cos_sin,
        problem.n_groups,
        problem.heads_per_group,
        problem.nope_dim,
        problem.rope_dim,
        quant_group_size,
        problem.is_neox,
    )


def call_fused_repeated(
    o: torch.Tensor,
    positions: torch.Tensor,
    rotary_cos_sin: torch.Tensor,
    problem: Problem,
    quant_group_size: int,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = None
    for _ in range(steps):
        out = call_fused(o, positions, rotary_cos_sin, problem, quant_group_size)
    assert out is not None
    return out


def bench_one(problem: Problem, args: argparse.Namespace) -> dict[str, float]:
    o, positions, rotary_cos_sin = make_inputs(problem, args)
    cudart = torch.cuda.cudart() if args.cuda_profiler_range else None

    if args.use_cuda_graph:
        # Build the allocator state before capture. The op allocates its
        # outputs internally, so capture a fixed repeated-call graph and replay
        # it. Capturing all benchmark steps would multiply peak graph-pool
        # memory; graph_capture_steps controls that tradeoff.
        for _ in range(min(args.warmup_steps, 5)):
            out = call_fused_repeated(
                o,
                positions,
                rotary_cos_sin,
                problem,
                args.quant_group_size,
                args.graph_capture_steps,
            )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = call_fused_repeated(
                o,
                positions,
                rotary_cos_sin,
                problem,
                args.quant_group_size,
                args.graph_capture_steps,
            )

        for _ in range(args.warmup_steps):
            graph.replay()
        torch.cuda.synchronize()

        repeat_us = []
        ops_per_repeat = args.bench_steps * args.graph_capture_steps
        if cudart is not None:
            cudart.cudaProfilerStart()
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.bench_steps):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            repeat_us.append(start.elapsed_time(end) * 1000.0 / ops_per_repeat)
        if cudart is not None:
            cudart.cudaProfilerStop()
    else:
        for _ in range(args.warmup_steps):
            out = call_fused(o, positions, rotary_cos_sin, problem, args.quant_group_size)
        torch.cuda.synchronize()

        repeat_us = []
        if cudart is not None:
            cudart.cudaProfilerStart()
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.bench_steps):
                out = call_fused(o, positions, rotary_cos_sin, problem, args.quant_group_size)
            end.record()
            torch.cuda.synchronize()
            repeat_us.append(start.elapsed_time(end) * 1000.0 / args.bench_steps)
        if cudart is not None:
            cudart.cudaProfilerStop()

    fp8_out, scale_out = out
    torch.cuda.synchronize()

    median_us = statistics.median(repeat_us)
    mean_us = statistics.fmean(repeat_us)
    min_us = min(repeat_us)
    max_us = max(repeat_us)

    result = {
        "median_us": median_us,
        "mean_us": mean_us,
        "min_us": min_us,
        "max_us": max_us,
        "fp8_numel": fp8_out.numel(),
        "scale_numel": scale_out.numel(),
        "graph_capture_steps": args.graph_capture_steps if args.use_cuda_graph else 0,
    }

    del out, fp8_out, scale_out, o, positions, rotary_cos_sin
    torch.cuda.empty_cache()
    return result


def validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise RuntimeError(
            f"Invalid CUDA device {args.device}; visible device count is {torch.cuda.device_count()}"
        )
    if args.warmup_steps < 0:
        raise RuntimeError("--warmup-steps must be >= 0")
    if args.bench_steps <= 0:
        raise RuntimeError("--bench-steps must be > 0")
    if args.repeats <= 0:
        raise RuntimeError("--repeats must be > 0")
    if args.graph_capture_steps <= 0:
        raise RuntimeError("--graph-capture-steps must be > 0")
    if args.quant_group_size != 128:
        raise RuntimeError("The CUDA op currently only supports --quant-group-size 128")
    if args.rope_dim != 64:
        raise RuntimeError("The CUDA op currently only supports --rope-dim 64")
    head_dim = args.nope_dim + args.rope_dim
    if head_dim % args.quant_group_size != 0:
        raise RuntimeError("head_dim must be a multiple of quant_group_size")
    chunks_per_head = head_dim // args.quant_group_size
    if chunks_per_head < 1 or chunks_per_head > 4:
        raise RuntimeError("head_dim / 128 must be in [1, 4]")
    expected_nope_mod = args.quant_group_size - args.rope_dim
    if args.nope_dim % args.quant_group_size != expected_nope_mod:
        raise RuntimeError(
            f"layout requires nope_dim % {args.quant_group_size} == {expected_nope_mod}; "
            f"got {args.nope_dim % args.quant_group_size}"
        )
    if not args.allow_non_sm100 and not is_sm_100f():
        raise RuntimeError(
            "This benchmark is intended for SM100. Pass --allow-non-sm100 to override."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.cuda.set_device(args.device)

    print("Benchmark: torch.ops.trtllm.fused_inv_rope_fp8_quant_vllm_port")
    print(f"device={torch.cuda.get_device_name(args.device)}")
    print(
        f"shape: n_groups={args.n_groups} heads_per_group={args.heads_per_group} "
        f"num_heads={args.n_groups * args.heads_per_group} "
        f"head_dim={args.nope_dim + args.rope_dim} nope_dim={args.nope_dim} "
        f"rope_dim={args.rope_dim} is_neox={args.is_neox} position_dtype={args.position_dtype}"
    )
    print(
        f"warmup_steps={args.warmup_steps} bench_steps={args.bench_steps} repeats={args.repeats} "
        f"use_cuda_graph={args.use_cuda_graph} graph_capture_steps="
        f"{args.graph_capture_steps if args.use_cuda_graph else 0} "
        f"cuda_profiler_range={args.cuda_profiler_range}"
    )
    print("")
    print("tokens,median_us,mean_us,min_us,max_us,fp8_numel,scale_numel,graph_capture_steps")

    for num_tokens in args.tokens:
        problem = Problem(
            num_tokens=num_tokens,
            n_groups=args.n_groups,
            heads_per_group=args.heads_per_group,
            nope_dim=args.nope_dim,
            rope_dim=args.rope_dim,
            is_neox=args.is_neox,
        )
        result = bench_one(problem, args)
        print(
            f"{num_tokens},"
            f"{result['median_us']:.3f},"
            f"{result['mean_us']:.3f},"
            f"{result['min_us']:.3f},"
            f"{result['max_us']:.3f},"
            f"{int(result['fp8_numel'])},"
            f"{int(result['scale_numel'])},"
            f"{int(result['graph_capture_steps'])}"
        )


if __name__ == "__main__":
    main()
