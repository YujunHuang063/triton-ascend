#!/usr/bin/env python3
"""
Dump the full TA (triton-ascend) IR pipeline of the FlashAttention forward
kernel ``fwd_kernel`` (from ``flash_attention_npu_v8_2.py``) all the way down
to the ``ttadapter`` (the linalg/MLIR that feeds ``bishengir-compile``).

Same idea / same method as ``dump_da_fwd_u_passes.py``:

    AST  --make_ttir-->  TTIR  --ttir_to_linalg-->  ttadapter

With ``MLIR_ENABLE_DUMP=1`` set before importing triton, every pass inside
both stages prints the IR before/after each pass via ``llvm::dbgs()``; those
per-pass dumps are captured to per-stage log files (OS-level fd-2 redirect),
and the final TTIR / ttadapter are saved as standalone ``.mlir`` files.

No NPU / CANN runtime is required: we stop at ``ttadapter`` (never invoke
``bishengir-compile``), so this runs purely host-side. The kernel's source
module imports ``torch_npu`` and queries device properties at import time, so
those NPU touchpoints are stubbed before the module is loaded.

Usage:
    .venv/bin/python third_party/ascend/unittest/pytest_ut/dump_fwd_kernel_passes.py
    .venv/bin/python third_party/ascend/unittest/pytest_ut/dump_fwd_kernel_passes.py --compact

Environment overrides:
    DUMP_OUT_DIR       output dir (default: /home/dev/work/workspace/ssbuf/test/fa/ir)
    DUMP_ARCH          target arch string (default: Ascend910_95)
    DUMP_ENABLE_SSBUF  "1" (default) reproduces the SSBUF / dynamic-CV-pipeline
                       ttadapter; "0" gives the plain ttadapter
    FA_SRC             dir containing flash_attention_npu_v8_2.py + utils.py
    DUMP_BLOCK_M       BLOCK_M constexpr (default: 64)
    DUMP_BLOCK_N       BLOCK_N constexpr (default: 64)
    DUMP_AICORE_NUM    AICORE_NUM constexpr / fake num_aicore (default: 40)
"""

import contextlib
import importlib.util
import os
import re
import sys
import types

# Must be set BEFORE any triton import (checked at pass-manager creation time).
os.environ.setdefault("MLIR_ENABLE_DUMP", "1")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler.compiler import ASTSource, make_backend  # noqa: E402
from triton._C.libtriton import ir, buffer_ir  # noqa: E402
from triton._C.libtriton.ascend import ir as ascend_ir  # noqa: E402

# The real compiler stage functions live in the ascend backend.
from triton.backends.ascend.compiler import make_ttir, ttir_to_linalg  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
#  Load the FlashAttention kernel module without a live NPU
# ═══════════════════════════════════════════════════════════════════════════

def _find_fa_src() -> str:
    env = os.environ.get("FA_SRC")
    if env:
        return env
    candidates = [
        "/home/dev/work/workspace/ssbuf/test/fa/src",
        "/home/dev/work/workspace/ssbuf支持/test/fa/src",
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "flash_attention_npu_v8_2.py")):
            return c
    raise FileNotFoundError(
        "Could not locate flash_attention_npu_v8_2.py; set FA_SRC env var."
    )


def _disable_autotune():
    """Neutralize @triton.autotune (it eagerly inits the runtime driver,
    which needs a real NPU). We pin BLOCK_M/BLOCK_N via CONSTANTS, so replace
    the decorator with a pass-through to the underlying @triton.jit function."""
    triton.autotune = lambda *args, **kwargs: (lambda fn: fn)


def _stub_npu():
    """Stub the NPU touchpoints the module hits at import time:
       - ``import torch_npu``
       - ``torch.npu.current_device()``
       - ``driver.active.utils.get_device_properties(device)``
    """
    if importlib.util.find_spec("torch_npu") is None:
        sys.modules["torch_npu"] = types.ModuleType("torch_npu")

    import torch
    if not hasattr(torch, "npu") or torch.npu is None:
        torch.npu = types.SimpleNamespace()
    torch.npu.current_device = lambda: 0

    import triton.runtime.driver as driver
    aicore = int(os.environ.get("DUMP_AICORE_NUM", "40"))
    vector = int(os.environ.get("DUMP_AICORE_NUM", "40"))
    fake_utils = types.SimpleNamespace(
        get_device_properties=lambda d: {
            "num_aicore": aicore,
            "num_vectorcore": vector,
        }
    )
    driver.active = types.SimpleNamespace(utils=fake_utils)


def load_kernel():
    _disable_autotune()
    _stub_npu()
    src = _find_fa_src()
    if src not in sys.path:
        sys.path.insert(0, src)  # so `from utils import is_hopper, is_ampere` resolves
    fa_path = os.path.join(src, "flash_attention_npu_v8_2.py")
    spec = importlib.util.spec_from_file_location("flash_attention_npu_v8_2", fa_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.fwd_kernel


# ═══════════════════════════════════════════════════════════════════════════
#  Signature / constants for fwd_kernel
#
#  Mirrors the production call in FlashAttentionFunc.forward with the __main__
#  test shapes (bs=8, seqlen=320, num_head=8, head_dim=64, bfloat16):
#      q,k,v:        (2560, 8, 64) bf16
#      o:            (2560, 8, 64) bf16   (q.new_empty)
#      l:            (2560, 8)     f32
#      q/k_attn_arg: (2560,)       i32
#      mask_tensor:  (8, 320, 320) bool
#      cu_seqlens_*: (9,)          i32
#      q_head=kv_head=8 (i32 scalars), scale=1/sqrt(64) (f32 scalar)
# ═══════════════════════════════════════════════════════════════════════════

BLOCK_M = int(os.environ.get("DUMP_BLOCK_M", "64"))
BLOCK_N = int(os.environ.get("DUMP_BLOCK_N", "64"))
AICORE_NUM = int(os.environ.get("DUMP_AICORE_NUM", "40"))

# Non-constexpr arguments -> {arg_name: triton_type_string}
SIGNATURE = {
    "q_ptr": "*bf16",
    "k_ptr": "*bf16",
    "v_ptr": "*bf16",
    "o_ptr": "*bf16",
    "l_ptr": "*fp32",
    "q_attn_arg_ptr": "*i32",
    "k_attn_arg_ptr": "*i32",
    "mask_tensor_ptr": "*i1",
    "cu_seqlens_q": "*i32",
    "cu_seqlens_k": "*i32",
    "q_head": "i32",
    "kv_head": "i32",
    "scale": "fp32",
}

# constexpr arguments -> {arg_name: value}
CONSTANTS = {
    "QK_DIM": 64,
    "V_DIM": 64,
    "MASK_FN": 1,
    "SPARSE_OPT": False,
    "DTYPE": 14,          # bf16 (19 would be fp16)
    "BLOCK_M": BLOCK_M,
    "BLOCK_N": BLOCK_N,
    "AICORE_NUM": AICORE_NUM,
    "MAX_Q_LEN": 320,
    "MAX_K_LEN": 320,
    "BATCH_SIZE": 8,
}


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers (identical to dump_da_fwd_u_passes.py)
# ═══════════════════════════════════════════════════════════════════════════

def strip_locations(ir_text: str) -> str:
    cleaned = re.sub(r'^#loc\d* = loc\(.+$', '', ir_text, flags=re.MULTILINE)
    cleaned = re.sub(r' loc\([^)]*\)', '', cleaned)
    cleaned = re.sub(r'^// -+.*-+ //\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def banner(title: str) -> None:
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


@contextlib.contextmanager
def capture_stderr_fd(path: str):
    """Redirect OS-level stderr (fd 2) to ``path`` so the MLIR pass-manager's
    ``llvm::dbgs()`` per-pass dumps are captured to file."""
    sys.stderr.flush()
    saved_fd = os.dup(2)
    log_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.dup2(log_fd, 2)
        os.close(log_fd)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, 2)
        os.close(saved_fd)


def write_text(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)
    print(f"  -> wrote {path}  ({len(text.splitlines())} lines)")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    compact = "-c" in sys.argv or "--compact" in sys.argv

    out_dir = os.environ.get(
        "DUMP_OUT_DIR", "/home/dev/work/workspace/ssbuf/test/fa/ir"
    )
    os.makedirs(out_dir, exist_ok=True)
    arch = os.environ.get("DUMP_ARCH", "Ascend910_95")
    enable_ssbuf = os.environ.get("DUMP_ENABLE_SSBUF", "1") == "1"

    kernel = load_kernel()
    # With autotune neutralized the kernel already *is* the JITFunction
    # (it has .arg_names); otherwise unwrap one level.
    fn = kernel if hasattr(kernel, "arg_names") else kernel.fn

    src = ASTSource(fn, SIGNATURE, CONSTANTS)

    target = GPUTarget("npu", arch, 32)
    backend = make_backend(target)
    options = backend.parse_options({
        "arch": arch,
        "compile_on_910_95": enable_ssbuf,
        "enable_dynamic_cv_pipeline": enable_ssbuf,
        "debug": False,
    })

    metadata = {
        "hash": "dump_fwd_kernel",
        "target": target,
        **options.__dict__,
    }

    context = ir.context()
    ir.load_dialects(context)
    buffer_ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    backend.load_dialects(context)
    codegen_fns = backend.get_codegen_implementation()
    module_map = backend.get_module_map()

    print(f"arch={arch}  enable_ssbuf={enable_ssbuf}  "
          f"BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N} AICORE_NUM={AICORE_NUM}")
    print(f"output dir: {out_dir}\n")

    # ── Phase 0: raw AST -> TTIR (before any optimization pass) ──
    banner("PHASE 0:  Initial TTIR (raw AST, before passes)")
    module = src.make_ir(options, codegen_fns, module_map, context)
    phase0 = str(module)
    if compact:
        phase0 = strip_locations(phase0)
    write_text(os.path.join(out_dir, "fwd_kernel.0_ast.ttir.mlir"),
               phase0 if compact else str(module))
    print()

    # ── Stage 1: make_ttir (per-pass dumps captured) ──
    banner("STAGE 1:  make_ttir  (AST-optimization passes, per-pass dump)")
    ttir_passes_log = os.path.join(out_dir, "stage1_ttir_passes.log")
    with capture_stderr_fd(ttir_passes_log):
        module = make_ttir(module, metadata, options)
    final_ttir = str(module)
    if compact:
        final_ttir = strip_locations(final_ttir)
    write_text(os.path.join(out_dir, "fwd_kernel.1_ttir.mlir"), final_ttir)
    print(f"  -> per-pass dump: {ttir_passes_log}")
    print()

    # ── Stage 2: ttir_to_linalg -> ttadapter (per-pass dumps captured) ──
    banner("STAGE 2:  ttir_to_linalg  (-> ttadapter, per-pass dump)")
    ttadapter_passes_log = os.path.join(out_dir, "stage2_ttadapter_passes.log")
    with capture_stderr_fd(ttadapter_passes_log):
        linalg = ttir_to_linalg(module, metadata, options, named_ops=True)
    if compact:
        linalg = strip_locations(linalg)
    write_text(os.path.join(out_dir, "fwd_kernel.2_ttadapter.mlir"), linalg)
    print(f"  -> per-pass dump: {ttadapter_passes_log}")
    print()

    banner("DONE")
    print(f"All IR written under: {out_dir}")
    print("  fwd_kernel.0_ast.ttir.mlir    initial TTIR from the Python AST")
    print("  stage1_ttir_passes.log        IR before/after each make_ttir pass")
    print("  fwd_kernel.1_ttir.mlir        optimized TTIR")
    print("  stage2_ttadapter_passes.log   IR before/after each lowering pass")
    print("  fwd_kernel.2_ttadapter.mlir   final ttadapter (input to bishengir-compile)")


if __name__ == "__main__":
    main()
