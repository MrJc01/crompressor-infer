#!/usr/bin/env python3
"""EXP-013B: Validação dos kernels C c1_matvec e c1_matmul_batch
contra a desquantização de referência em PyTorch na camada L11/E0.
"""
import ctypes
import os
import struct
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
LIB_SO = ROOT / "engine" / "libcrom_infer.so"
HYBRID_MODEL = ROOT / "artifacts" / "granite-1b-hybrid.cromllm"

class CromC1Mat(ctypes.Structure):
    _fields_ = [
        ("codebook", ctypes.c_void_p),
        ("codes", ctypes.c_void_p),
        ("deltas", ctypes.c_void_p),
        ("scales", ctypes.c_void_p),
        ("rows", ctypes.c_int),
        ("cols", ctypes.c_int),
        ("k_delta", ctypes.c_int),
    ]

def main():
    print(f"=== Validação do Kernel C1_TOPK_K8 (EXP-013B) ===")
    if not LIB_SO.exists():
        print(f"ERRO: {LIB_SO} não encontrado. Execute 'make all'.")
        sys.exit(1)
    if not HYBRID_MODEL.exists():
        print(f"ERRO: {HYBRID_MODEL} não encontrado.")
        sys.exit(1)

    lib = ctypes.CDLL(str(LIB_SO))
    lib.c1_mat_from.restype = CromC1Mat
    lib.c1_mat_from.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

    lib.c1_matvec.restype = None
    lib.c1_matvec.argtypes = [ctypes.POINTER(CromC1Mat), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

    lib.c1_matmul_batch.restype = None
    lib.c1_matmul_batch.argtypes = [
        ctypes.POINTER(CromC1Mat), ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int
    ]

    # Carrega diretório e payload de L11/E0
    with open(HYBRID_MODEL, "rb") as f:
        hdr = f.read(144)
        dir_off = struct.unpack_from("<I", hdr, 104)[0]
        f.seek(dir_off + (11 * 32 + 0) * 24)
        e_off, e_sz, fmt, _ = struct.unpack("<QQII", f.read(24))
        print(f"L11/E0 no arquivo: offset={e_off}, size={e_sz} B, format_id={fmt}")
        assert fmt == 2, f"Esperado format_id=2 (C1_TOPK_K8), obtido {fmt}"
        assert e_sz == 696320, f"Esperado 696.320 bytes, obtido {e_sz}"

        f.seek(e_off)
        blob = bytearray(f.read(e_sz))

    sys.path.insert(0, str(ROOT))
    from tools.pack_cromllm import c1_topk_dequant
    in_bytes = 462848
    out_bytes = 233472
    blob_in = blob[:in_bytes]
    blob_out = blob[in_bytes:in_bytes + out_bytes]

    # Reconstrução PyTorch exata
    print("Desquantizando matrizes de L11/E0 em PyTorch...")
    W_in_ref = c1_topk_dequant(blob_in, 1024, 1024, 8)
    W_out_ref = c1_topk_dequant(blob_out, 1024, 512, 8)

    c_blob_in = (ctypes.c_uint8 * len(blob_in)).from_buffer(blob_in)
    c_blob_out = (ctypes.c_uint8 * len(blob_out)).from_buffer(blob_out)

    mat_in = lib.c1_mat_from(ctypes.cast(c_blob_in, ctypes.c_void_p), 1024, 1024, 8)
    mat_out = lib.c1_mat_from(ctypes.cast(c_blob_out, ctypes.c_void_p), 1024, 512, 8)

    # 1. Teste c1_matvec em input_linear [1024, 1024]
    torch.manual_seed(1234)
    x1 = torch.randn(1024, dtype=torch.float32)
    y1_ref = W_in_ref @ x1
    y1_c = torch.zeros(1024, dtype=torch.float32)

    lib.c1_matvec(
        ctypes.byref(mat_in),
        ctypes.c_void_p(x1.data_ptr()),
        ctypes.c_void_p(y1_c.data_ptr()),
        6
    )

    err1 = (y1_c - y1_ref).norm() / y1_ref.norm()
    print(f"[TEST 1] c1_matvec input_linear [1024, 1024]:")
    print(f"         L2 rel error = {err1.item():.4e}")
    assert err1 < 1e-4, f"Erro L2 {err1} excede tolerância 1e-4"
    print("         -> PASS!")

    # 2. Teste c1_matvec em output_linear [1024, 512]
    x2 = torch.randn(512, dtype=torch.float32)
    y2_ref = W_out_ref @ x2
    y2_c = torch.zeros(1024, dtype=torch.float32)

    lib.c1_matvec(
        ctypes.byref(mat_out),
        ctypes.c_void_p(x2.data_ptr()),
        ctypes.c_void_p(y2_c.data_ptr()),
        6
    )

    err2 = (y2_c - y2_ref).norm() / y2_ref.norm()
    print(f"[TEST 2] c1_matvec output_linear [1024, 512]:")
    print(f"         L2 rel error = {err2.item():.4e}")
    assert err2 < 1e-4, f"Erro L2 {err2} excede tolerância 1e-4"
    print("         -> PASS!")

    # 3. Teste c1_matmul_batch em input_linear com B=4 tokens
    B = 4
    X = torch.randn(B, 1024, dtype=torch.float32)
    Y_ref = X @ W_in_ref.T
    Y_c = torch.zeros(B, 1024, dtype=torch.float32)

    lib.c1_matmul_batch(
        ctypes.byref(mat_in),
        ctypes.c_void_p(X.data_ptr()),
        B,
        1024,
        ctypes.c_void_p(Y_c.data_ptr()),
        1024,
        6
    )

    err3 = (Y_c - Y_ref).norm() / Y_ref.norm()
    print(f"[TEST 3] c1_matmul_batch input_linear [B={B}, 1024] x [1024, 1024]^T:")
    print(f"         L2 rel error = {err3.item():.4e}")
    assert err3 < 1e-4, f"Erro L2 {err3} excede tolerância 1e-4"
    print("         -> PASS!")

    # 4. Teste c1_matmul_batch em output_linear com B=7 tokens
    B7 = 7
    X7 = torch.randn(B7, 512, dtype=torch.float32)
    Y7_ref = X7 @ W_out_ref.T
    Y7_c = torch.zeros(B7, 1024, dtype=torch.float32)

    lib.c1_matmul_batch(
        ctypes.byref(mat_out),
        ctypes.c_void_p(X7.data_ptr()),
        B7,
        512,
        ctypes.c_void_p(Y7_c.data_ptr()),
        1024,
        6
    )

    err4 = (Y7_c - Y7_ref).norm() / Y7_ref.norm()
    print(f"[TEST 4] c1_matmul_batch output_linear [B={B7}, 512] x [1024, 512]^T:")
    print(f"         L2 rel error = {err4.item():.4e}")
    assert err4 < 1e-4, f"Erro L2 {err4} excede tolerância 1e-4"
    print("         -> PASS!")

    print(f"\nTODOS OS 4 TESTES PASSARAM! (Max L2 error = {max(err1, err2, err3, err4):.4e} < 1e-4)")

if __name__ == "__main__":
    main()
