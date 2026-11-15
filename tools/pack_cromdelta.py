#!/usr/bin/env python3
"""EXP-014: Empacotador de Delta de Fine-Tune .cromdelta
Gera artifacts/granite-1b-instruct.cromdelta a partir da diferença
entre ibm-granite/granite-3.1-1b-a400m-instruct e ibm-granite/granite-3.1-1b-a400m-base.

Layout do .cromdelta:
1. Header 4 KiB (magic "CROMLD01", version, dimensões, offsets)
2. tokenizer.json (embutido do Instruct)
3. embed_tokens [49155, 1024] FP16 (tabela completa do Instruct)
4. norms FP32 (49 vetores de 1024)
5. routers FP32 (24 x 32 x 1024)
6. attn_deltas 1-bit por grupo de 64 (24 camadas x (q, k, v, o))
7. expert_dir (768 entradas de 24 B)
8. expert_deltas 1-bit por grupo de 64 (768 experts x (input_linear, output_linear))
"""
from __future__ import annotations

import os
import struct
import sys
import time
from pathlib import Path
import torch
from safetensors import safe_open
from transformers.utils.hub import cached_file

ROOT = Path(__file__).resolve().parents[1]
PAGE = 4096
MAGIC = b"CROMLD01"
VERSION = 1
DELTA_1BIT = 1
G = 64

# Header: 8s + 13I + 10Q + 2I + 2Q + padding a 4096
# magic(8s), version(I), delta_fmt(I), vocab_base(I), vocab_delta(I),
# hidden(I), inter(I), layers(I), q_heads(I), kv_heads(I), head_dim(I),
# num_exp(I), group_size(I), reserved(I), -> 13 uint32
# tok_off(Q), tok_sz(Q), embed_off(Q), embed_sz(Q), norms_off(Q), norms_sz(Q),
# routers_off(Q), routers_sz(Q), attn_off(Q), attn_sz(Q), -> 10 uint64
# dir_off(Q), dir_count(I), dir_entry_sz(I), exp_off(Q), exp_sz(Q) -> 1Q, 2I, 2Q = total 11Q + 2I
HEADER_FMT = "<8s13I10QQIIQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def pack_1bit_matrix(diff: torch.Tensor) -> bytes:
    """Empacota matriz diff [rows, cols] em 1 bit por peso por grupo de 64.
    Para cada grupo de 64:
      - 8 bytes de sinais (bit i = 1 se diff >= 0, 0 se diff < 0)
      - 2 bytes de escala FP16 alpha = mean(|diff|)
    Retorna bytes: (signs contíguos de todas as linhas) + (scales contíguas).
    """
    rows, cols = diff.shape
    ng = cols // G
    diff_g = diff.float().view(-1, G) # [num_groups, 64]
    alpha = diff_g.abs().mean(dim=1, keepdim=True).half() # [num_groups, 1] fp16
    
    # Empacota 64 bits em 8 bytes por grupo
    # diff_g >= 0 -> 1, diff_g < 0 -> 0
    bits = (diff_g >= 0).to(torch.uint8).view(-1, 8, 8) # [num_groups, 8 bytes, 8 bits]
    # Converte 8 bits em 1 byte: sum(bit_j << j)
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)
    signs_bytes = (bits * powers).sum(dim=-1, dtype=torch.uint8) # [num_groups, 8]

    signs_raw = signs_bytes.numpy().tobytes()
    scales_raw = alpha.numpy().tobytes()
    return signs_raw + scales_raw


def dequant_1bit_matrix(blob: bytes, rows: int, cols: int) -> torch.Tensor:
    """Decodifica matriz de 1-bit de volta para float32 para validação."""
    ng = cols // G
    num_groups = rows * ng
    signs_size = num_groups * 8
    scales_size = num_groups * 2

    signs_bytes = torch.frombuffer(bytearray(blob[:signs_size]), dtype=torch.uint8).view(num_groups, 8)
    scales = torch.frombuffer(bytearray(blob[signs_size:signs_size + scales_size]), dtype=torch.float16).float().view(num_groups, 1)

    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)
    # expande 8 bytes em 64 bits
    bits = ((signs_bytes.unsqueeze(-1) & powers) > 0).float() # [num_groups, 8, 8]
    signs = torch.where(bits > 0, 1.0, -1.0).view(num_groups, 64)
    recon = (signs * scales).view(rows, cols)
    return recon


def mat_delta_1bit_bytes(rows: int, cols: int) -> int:
    ng = cols // G
    num_groups = rows * ng
    return num_groups * 8 + num_groups * 2


def align4k(pos: int) -> int:
    return (pos + PAGE - 1) & ~(PAGE - 1)


def section(f, pos: int, data: bytes) -> int:
    f.write(data)
    pos += len(data)
    if pos % PAGE:
        pad = PAGE - (pos % PAGE)
        f.write(b"\x00" * pad)
        pos += pad
    return pos


def pack_delta(
    base_id="ibm-granite/granite-3.1-1b-a400m-base",
    inst_id="ibm-granite/granite-3.1-1b-a400m-instruct",
    out_path="artifacts/granite-1b-instruct.cromdelta",
):
    print(f"=== Empacotando Delta: {inst_id} vs {base_id} ===")
    t0 = time.time()
    torch.set_num_threads(6)

    # 1. Localiza os safetensors de ambos os modelos no cache HF
    from huggingface_hub import scan_cache_dir
    cache = scan_cache_dir()

    base_shards = []
    inst_path = None
    for r in cache.repos:
        if r.repo_id == BASE_ID:
            for rev in r.revisions:
                base_shards = [os.path.join(rev.snapshot_path, f) for f in os.listdir(rev.snapshot_path) if f.endswith(".safetensors")]
        elif r.repo_id == INST_ID:
            for rev in r.revisions:
                candidate = os.path.join(rev.snapshot_path, "model.safetensors")
                if os.path.exists(candidate):
                    inst_path = candidate

    assert inst_path, f"model.safetensors do Instruct ({inst_id}) não encontrado!"
    assert base_shards, f"shards do Base ({base_id}) não encontrados!"

    base_map = {}
    for shard in base_shards:
        with safe_open(shard, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                base_map[k] = shard

    def get_base(k):
        with safe_open(base_map[k], framework="pt", device="cpu") as sf:
            return sf.get_tensor(k)

    # 2. Tokenizer do Instruct
    tok_file = cached_file(inst_id, "tokenizer.json", local_files_only=True)
    with open(tok_file, "rb") as f:
        tok_bytes = f.read()
    print(f"tokenizer.json do Instruct: {len(tok_bytes):,} bytes")

    # 3. Cria arquivo de saída
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    f = open(out_path, "wb")
    f.seek(PAGE) # reserva 4096 bytes para o header
    cur_pos = PAGE

    # Seção 1: Tokenizer
    tok_off = cur_pos
    tok_sz = len(tok_bytes)
    cur_pos = section(f, cur_pos, tok_bytes)
    print(f"tokenizer gravado no offset {tok_off}")

    # Seção 2: Embed tokens completo do Instruct [49155, 1024] em FP16
    with safe_open(inst_path, framework="pt", device="cpu") as sf:
        embed_inst = sf.get_tensor("model.embed_tokens.weight").half()
    embed_bytes = embed_inst.numpy().tobytes()
    embed_off = cur_pos
    embed_sz = len(embed_bytes)
    cur_pos = section(f, cur_pos, embed_bytes)
    print(f"embed_tokens gravado no offset {embed_off} ({embed_sz:,} B, shape {embed_inst.shape})")
    del embed_inst, embed_bytes

    # Seção 3: Norms FP32 do Instruct (49 vetores de 1024)
    norms_list = []
    with safe_open(inst_path, framework="pt", device="cpu") as sf:
        for li in range(24):
            norms_list.append(sf.get_tensor(f"model.layers.{li}.input_layernorm.weight").float())
            norms_list.append(sf.get_tensor(f"model.layers.{li}.post_attention_layernorm.weight").float())
        norms_list.append(sf.get_tensor("model.norm.weight").float())
    norms_tensor = torch.cat(norms_list, dim=0) # [49 * 1024]
    norms_bytes = norms_tensor.numpy().tobytes()
    norms_off = cur_pos
    norms_sz = len(norms_bytes)
    cur_pos = section(f, cur_pos, norms_bytes)
    print(f"norms gravadas no offset {norms_off} ({norms_sz:,} B)")
    del norms_list, norms_tensor, norms_bytes

    # Seção 4: Routers FP32 do Instruct (24 camadas x 32 experts x 1024)
    routers_list = []
    with safe_open(inst_path, framework="pt", device="cpu") as sf:
        for li in range(24):
            routers_list.append(sf.get_tensor(f"model.layers.{li}.block_sparse_moe.router.layer.weight").float())
    routers_tensor = torch.cat([r.view(-1) for r in routers_list], dim=0)
    routers_bytes = routers_tensor.numpy().tobytes()
    routers_off = cur_pos
    routers_sz = len(routers_bytes)
    cur_pos = section(f, cur_pos, routers_bytes)
    print(f"routers gravados no offset {routers_off} ({routers_sz:,} B)")
    del routers_list, routers_tensor, routers_bytes

    # Seção 5: Atenção 1-bit deltas (24 camadas x (q, k, v, o))
    print("Empacotando deltas de atenção em 1-bit...")
    attn_bytes_list = []
    with safe_open(inst_path, framework="pt", device="cpu") as sf:
        for li in range(24):
            for proj, (rows, cols) in [
                ("q_proj", (1024, 1024)),
                ("k_proj", (512, 1024)),
                ("v_proj", (512, 1024)),
                ("o_proj", (1024, 1024)),
            ]:
                name = f"model.layers.{li}.self_attn.{proj}.weight"
                w_inst = sf.get_tensor(name)
                w_base = get_base(name)
                diff = w_inst - w_base
                b_packed = pack_1bit_matrix(diff)
                assert len(b_packed) == mat_delta_1bit_bytes(rows, cols)
                attn_bytes_list.append(b_packed)
                del w_inst, w_base, diff

    attn_all_bytes = b"".join(attn_bytes_list)
    attn_off = cur_pos
    attn_sz = len(attn_all_bytes)
    cur_pos = section(f, cur_pos, attn_all_bytes)
    print(f"atencao deltas gravados no offset {attn_off} ({attn_sz:,} B)")
    del attn_bytes_list, attn_all_bytes

    # Seção 6 e 7: Diretório de Experts e Deltas de Experts (768 experts)
    # Tamanho de cada expert delta:
    # input_linear [1024, 1024]: 163.840 B + output_linear [1024, 512]: 81.920 B = 245.760 B (60 páginas 4K)
    exp_delta_sz = mat_delta_1bit_bytes(1024, 1024) + mat_delta_1bit_bytes(1024, 512)
    assert exp_delta_sz == 245760
    assert exp_delta_sz % PAGE == 0, "exp_delta_sz deve ser múltiplo de 4096!"

    dir_count = 24 * 32
    dir_off = cur_pos
    dir_entry_sz = 24
    dir_sz = dir_count * dir_entry_sz
    cur_pos = align4k(cur_pos + dir_sz) # alinha para início dos deltas de experts
    exp_off = cur_pos

    print(f"Gravando {dir_count} experts deltas a partir do offset {exp_off}...")
    dir_entries = []
    exp_cursor = exp_off

    with safe_open(inst_path, framework="pt", device="cpu") as sf:
        for li in range(24):
            in_name = f"model.layers.{li}.block_sparse_moe.input_linear.weight"
            out_name = f"model.layers.{li}.block_sparse_moe.output_linear.weight"
            il_inst = sf.get_tensor(in_name)
            il_base = get_base(in_name)
            ol_inst = sf.get_tensor(out_name)
            ol_base = get_base(out_name)

            for ei in range(32):
                b_in = pack_1bit_matrix(il_inst[ei] - il_base[ei])
                b_out = pack_1bit_matrix(ol_inst[ei] - ol_base[ei])

                exp_blob = b_in + b_out
                assert len(exp_blob) == exp_delta_sz

                f.seek(exp_cursor)
                f.write(exp_blob)

                # Entrada de diretório: offset, size, format_id (1=1-bit), reserved
                dir_entries.append(struct.pack("<QQII", exp_cursor, exp_delta_sz, DELTA_1BIT, 0))
                exp_cursor += exp_delta_sz
            del il_inst, il_base, ol_inst, ol_base

    # Grava o diretório de experts no dir_off
    f.seek(dir_off)
    f.write(b"".join(dir_entries))
    total_exp_sz = exp_cursor - exp_off
    print(f"experts deltas gravados ({total_exp_sz:,} B, {dir_count} entradas alinhadas a 4K)")

    # Seção 0: Grava Header em 0
    header_vals = (
        MAGIC,
        VERSION,
        DELTA_1BIT,
        49152,  # vocab_base
        49155,  # vocab_delta
        1024,   # hidden
        512,    # inter
        24,     # layers
        16,     # q_heads
        8,      # kv_heads
        64,     # head_dim
        32,     # num_exp
        64,     # group_size
        0,      # reserved
        tok_off, tok_sz,
        embed_off, embed_sz,
        norms_off, norms_sz,
        routers_off, routers_sz,
        attn_off, attn_sz,
        dir_off, dir_count, dir_entry_sz,
        exp_off, total_exp_sz,
    )
    header_bytes = struct.pack(HEADER_FMT, *header_vals)
    header_padded = header_bytes + b"\x00" * (PAGE - len(header_bytes))

    f.seek(0)
    f.write(header_padded)
    f.close()

    total_file_bytes = exp_cursor
    print(f"\nArquivo .cromdelta concluído com sucesso!")
    print(f"Caminho: {out_path}")
    print(f"Tamanho total: {total_file_bytes:,} bytes ({total_file_bytes / 1e6:.1f} MB / {total_file_bytes / (1024*1024):.1f} MiB)")
    print(f"Tempo de empacotamento: {time.time() - t0:.2f}s")

    # Round-trip check rápido em L0.o_proj e L11/E0
    print("\nExecutando round-trip check de integridade no .cromdelta...")
    with open(out_path, "rb") as rf:
        hdr = rf.read(PAGE)
        assert hdr[:8] == MAGIC, "Magic inválido!"
        
        # Lê delta de L0.o_proj
        # Atenção L0: q, k, v, o
        q_sz = mat_delta_1bit_bytes(1024, 1024)
        k_sz = mat_delta_1bit_bytes(512, 1024)
        v_sz = mat_delta_1bit_bytes(512, 1024)
        o_off = attn_off + q_sz + k_sz + v_sz
        o_sz = mat_delta_1bit_bytes(1024, 1024)
        rf.seek(o_off)
        blob_o = rf.read(o_sz)
        delta_o_recon = dequant_1bit_matrix(blob_o, 1024, 1024)

        with safe_open(inst_path, framework="pt", device="cpu") as sf:
            w_o_inst = sf.get_tensor("model.layers.0.self_attn.o_proj.weight")
            w_o_base = get_base("model.layers.0.self_attn.o_proj.weight")
            diff_o = (w_o_inst - w_o_base).float()
            err_o = (delta_o_recon - diff_o).norm() / diff_o.norm()
            print(f"Round-trip L0.o_proj delta: erro L2 = {err_o.item():.4f} (esperado ~0.2-0.3 para 1-bit)")

        # Lê delta de L11/E0
        dir_entry_off = dir_off + (11 * 32 + 0) * 24
        rf.seek(dir_entry_off)
        e_off, e_sz, e_fmt, _ = struct.unpack("<QQII", rf.read(24))
        rf.seek(e_off)
        e_blob = rf.read(e_sz)
        in_delta_sz = mat_delta_1bit_bytes(1024, 1024)
        delta_in_recon = dequant_1bit_matrix(e_blob[:in_delta_sz], 1024, 1024)

        with safe_open(inst_path, framework="pt", device="cpu") as sf:
            w_in_inst = sf.get_tensor("model.layers.11.block_sparse_moe.input_linear.weight")[0]
            w_in_base = get_base("model.layers.11.block_sparse_moe.input_linear.weight")[0]
            diff_in = (w_in_inst - w_in_base).float()
            err_in = (delta_in_recon - diff_in).norm() / diff_in.norm()
            print(f"Round-trip L11/E0 input_linear delta: erro L2 = {err_in.item():.4f}")

    print("Round-trip check: PASS / OK!\n")


if __name__ == "__main__":
    BASE_ID = "ibm-granite/granite-3.1-1b-a400m-base"
    INST_ID = "ibm-granite/granite-3.1-1b-a400m-instruct"
    pack_delta(BASE_ID, INST_ID, "artifacts/granite-1b-instruct.cromdelta")
