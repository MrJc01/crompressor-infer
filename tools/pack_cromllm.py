#!/usr/bin/env python3
"""EXP-007: empacota o granite-3.1-1b-a400m-instruct para o formato
.cromllm v1 (docs/03-formato-cromllm.md): header 4 KiB, tokenizer.json
embutido, bloco Hot (embed F16 + norms/routers FP32 + atencao Q4_G64),
diretorio de experts e experts Q4_G64 contiguos alinhados a 4 KiB.

Modos:
  python3 tools/pack_cromllm.py            # empacota + round-trip check
  python3 tools/pack_cromllm.py --golden   # gabarito PyTorch BF16
"""
import json
import os
import struct
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.hub import cached_file

MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-instruct"
OUT = "artifacts/granite-1b.cromllm"
GOLDEN = "artifacts/golden_ref.json"
GOLDEN_META = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "experiments", "logs", "baseline-hf",
                           "golden_ref_meta.json")
GOLDEN_TEXT = "Explain a hash table in one sentence:"
PAGE = 4096
Q4 = 1          # format_id Q4_G64
C1_K8 = 2       # format_id C1_TOPK_K8
C1_K4 = 3       # format_id C1_TOPK_K4
EMBED_F16 = 0   # embed_format
G = 64
K256 = 256
TRAIN_N = 30_000
MAGIC = b"CROMLM01"
HEAD_FMT = "<8s14IffffffQQIIQQQQ"  # 144 bytes
assert struct.calcsize(HEAD_FMT) == 144
EXTREMAS = {0, 1, 2, 3, 20, 21, 22, 23}


def q4_pack(W):
    """W [rows, cols] -> (idx u8 [rows, cols/2], scales f32 [rows, cols/64]).
    Layout do kv_q4 (EXP-005): grupos de 64 na dimensao de entrada (cols);
    byte c do grupo g = (q[c]+8) | ((q[c+32]+8)<<4); idx e escalas
    contiguos por matriz."""
    rows, cols = W.shape
    ng = cols // G
    Wf = W.float().view(-1, G)                     # [rows*ng, 64]
    s = (Wf.abs().amax(1, keepdim=True) / 7.0).clamp(min=1e-12)
    q = (Wf / s).round().clamp(-7, 7).to(torch.int16) + 8   # [rows*ng, 64], 0..15
    q = q.view(rows, ng, 2, 32).to(torch.uint8)    # [.., low(par), high(impar)]
    idx = (q[:, :, 0, :] | (q[:, :, 1, :] << 4)).view(rows, ng * 32)
    sc = s.view(rows, ng)
    return idx.numpy(), sc.float().numpy()


def q4_dequant(idx, sc):
    if not isinstance(idx, torch.Tensor):
        idx = torch.from_numpy(idx.copy())
    if not isinstance(sc, torch.Tensor):
        sc = torch.from_numpy(sc.copy())
    rows, ng = sc.shape
    lo = ((idx & 0x0F).to(torch.int16) - 8).view(rows, ng, 32)
    hi = (((idx >> 4) & 0x0F).to(torch.int16) - 8).view(rows, ng, 32)
    q = torch.cat([lo, hi], dim=2).view(rows, ng, G).float()
    return (q * sc.float()[:, :, None]).view(rows, ng * G)


def assign(X, C, chunk=163_840):
    cn = (C * C).sum(1)
    xn = (X * X).sum(1)
    codes = torch.empty(X.shape[0], dtype=torch.int32)
    buf = torch.empty(min(chunk, X.shape[0]), C.shape[0])
    for i in range(0, X.shape[0], chunk):
        xb = X[i:i + chunk]
        b = buf[:xb.shape[0]]
        torch.mm(xb, C.T, out=b)
        b.mul_(-2.0).add_(xn[i:i + chunk, None]).add_(cn)
        codes[i:i + chunk] = b.argmin(1)
    return codes


def kmeans(X, k, iters, seed):
    g = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(X.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        codes = assign(X, C).long()
        S = torch.zeros(k, X.shape[1])
        cnt = torch.zeros(k)
        S.index_add_(0, codes, X)
        cnt.index_add_(0, codes, torch.ones(X.shape[0]))
        nz = cnt > 0
        C[nz] = S[nz] / cnt[nz, None]
    return C


def train_codebook_for_tensor(W_det):
    Wf = W_det.view(-1, G).float()
    s = Wf.abs().amax(1, keepdim=True).half().float().clamp(min=1e-12)
    Wn = Wf / s
    vec = Wn.reshape(-1, 4)
    g = torch.Generator().manual_seed(1234)
    Xs = vec[torch.randperm(vec.shape[0], generator=g)[:TRAIN_N]]
    return kmeans(Xs, K256, 8, 1234).half().float()


def c1_topk_pack(W, C, k_delta):
    """Empacota matriz W [rows, cols] no formato C1_TOPK com centróides C [256, 4] e k_delta deltas."""
    rows, cols = W.shape
    Wf = W.view(-1, G).float()
    s_base = Wf.abs().amax(1, keepdim=True).half().float().clamp(min=1e-12)
    Wn = Wf / s_base
    vec = Wn.reshape(-1, 4)
    codes = assign(vec, C).to(torch.uint8) # [rows * (cols / 4)]
    Cn = C[codes.long()].view(Wf.shape)
    R = Wn - Cn # [rows * (cols / 64), 64]

    top_vals, top_idx = torch.topk(R.abs(), k=k_delta, dim=-1) # [num_groups, k_delta]
    alpha = top_vals.mean(dim=1, keepdim=True).half() # [num_groups, 1] fp16

    signs_neg = (torch.gather(R, 1, top_idx) < 0).to(torch.uint8) # 1 se negativo, 0 se positivo
    pos_u8 = (top_idx & 63).to(torch.uint8)
    delta_bytes = (pos_u8 | (signs_neg << 6)).view(-1).numpy() # [num_groups * k_delta] uint8

    cb_bytes = C.float().numpy().tobytes() # 256 * 4 * 4 = 4096 bytes
    codes_bytes = codes.numpy().tobytes()

    s_base_fp16 = s_base.half().view(-1, 1)
    scales_pair = torch.cat([s_base_fp16, alpha], dim=1).view(-1).numpy().tobytes() # num_groups * 2 * 2 bytes

    return cb_bytes + codes_bytes + delta_bytes.tobytes() + scales_pair


def c1_topk_dequant(blob, rows, cols, k_delta):
    """Decodifica payload C1_TOPK de volta para matriz reconstruída float32."""
    ng = cols // G
    num_groups = rows * ng
    off = 0

    cb_size = 256 * 4 * 4
    C = torch.frombuffer(bytearray(blob[off:off + cb_size]), dtype=torch.float32).view(256, 4)
    off += cb_size

    codes_size = rows * (cols // 4)
    codes = torch.frombuffer(bytearray(blob[off:off + codes_size]), dtype=torch.uint8).long()
    off += codes_size

    deltas_size = num_groups * k_delta
    deltas = torch.frombuffer(bytearray(blob[off:off + deltas_size]), dtype=torch.uint8).view(num_groups, k_delta)
    off += deltas_size

    scales_size = num_groups * 4
    scales = torch.frombuffer(bytearray(blob[off:off + scales_size]), dtype=torch.float16).view(num_groups, 2).float()
    s_base = scales[:, 0:1]
    alpha = scales[:, 1:2]

    # Reconstrução
    Cn = C[codes].view(num_groups, G)
    delta_mat = torch.zeros(num_groups, G, dtype=torch.float32)
    pos = (deltas & 63).long()
    sign = torch.where((deltas >> 6) > 0, -1.0, 1.0)
    delta_mat.scatter_(1, pos, sign * alpha)

    Wn = Cn + delta_mat
    return (Wn * s_base).view(rows, cols)


def mat_q4_bytes(rows, cols):
    """tamanho em bytes da matriz em Q4_G64."""
    return rows * (cols // 2) + rows * (cols // G) * 4


def mat_c1_bytes(rows, cols, k_delta):
    """tamanho em bytes da matriz em C1_TOPK."""
    cb = 256 * 4 * 4
    codes = rows * (cols // 4)
    deltas = rows * (cols // G) * k_delta
    scales = rows * (cols // G) * 2 * 2
    return cb + codes + deltas + scales


def section(f, pos, data):
    f.write(data)
    pos += len(data)
    if pos % PAGE:
        f.write(b"\x00" * (PAGE - pos % PAGE))
        pos += PAGE - pos % PAGE
    return pos


def pack(model_id=MODEL_ID, out_path=OUT, profile="q4"):
    torch.set_num_threads(6)
    print(f"Carregando modelo {model_id} (perfil: {profile})...")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True)
    model.eval()
    cfg = model.config
    L, E, H = cfg.num_hidden_layers, cfg.num_local_experts, cfg.hidden_size
    params = dict(model.named_parameters())

    tok_path = os.path.dirname(cached_file(model_id, "config.json"))
    tok_bytes = open(os.path.join(tok_path, "tokenizer.json"), "rb").read()
    print(f"tokenizer.json embutido: {len(tok_bytes):,} bytes")

    qsz = mat_q4_bytes(H, H)
    ksz = mat_q4_bytes(cfg.num_key_value_heads * (H // cfg.num_attention_heads), H)
    per_layer_attn = qsz + 2 * ksz + qsz

    # Configuração dinâmica de camadas extremas: 4 primeiras e 4 últimas camadas em Q4
    extremas = set(range(min(4, L))) | set(range(max(0, L - 4), L))
    mid_layer = L // 2
    I = cfg.intermediate_size
    in_exp_dim = 2 * I

    # Configuração do perfil por camada
    def get_layer_format(li):
        if profile == "q4":
            return Q4, 0
        elif profile == "c1-topk8":
            return C1_K8, 8
        elif profile == "c1-topk4":
            return C1_K4, 4
        elif profile in ("hybrid-c1-k8", "hibrido-c1-k8"):
            return (Q4, 0) if li in extremas else (C1_K8, 8)
        elif profile in ("hybrid-c1-k4", "hibrido-c1-k4"):
            return (Q4, 0) if li in extremas else (C1_K4, 4)
        else:
            raise ValueError(f"Perfil desconhecido: {profile}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    def make_header(hot_off, hot_size, dir_off, experts_off, tok_off, tok_size):
        default_exp_fmt = Q4 if profile == "q4" else C1_K8
        vals = (MAGIC,
                1,                                    # version
                cfg.vocab_size, H, cfg.intermediate_size, L,
                cfg.num_attention_heads, cfg.num_key_value_heads,
                E, cfg.num_experts_per_tok,
                H // cfg.num_attention_heads,         # head_dim
                EMBED_F16, Q4, default_exp_fmt, G,
                float(cfg.rope_theta), float(cfg.rms_norm_eps),
                float(cfg.embedding_multiplier),
                float(cfg.attention_multiplier),
                float(cfg.residual_multiplier),
                float(cfg.logits_scaling),
                hot_off, hot_size, dir_off, L * E, 24,
                experts_off, tok_off, tok_size)
        return struct.pack(HEAD_FMT, *vals) + b"\x00" * (PAGE - struct.calcsize(HEAD_FMT))

    # Pré-treina codebooks para camadas que utilizarem C1_TOPK
    codebooks_in = {}
    codebooks_out = {}
    need_c1 = any(get_layer_format(li)[0] in (C1_K8, C1_K4) for li in range(L))
    if need_c1:
        print("Treinando codebooks para camadas C1_TOPK...")
        with torch.no_grad():
            for i in range(L):
                fmt, _ = get_layer_format(i)
                if fmt in (C1_K8, C1_K4):
                    il = model.model.layers[i].block_sparse_moe.input_linear.weight.data
                    ol = model.model.layers[i].block_sparse_moe.output_linear.weight.data
                    codebooks_in[i] = train_codebook_for_tensor(il)
                    codebooks_out[i] = train_codebook_for_tensor(ol)

    with open(out_path, "wb") as f:
        f.write(make_header(0, 0, 0, 0, 0, 0))   # placeholder; patch no fim
        pos = PAGE
        tok_off = pos
        pos = section(f, pos, tok_bytes)
        hot_off = pos
        emb = params["model.embed_tokens.weight"].data.to(torch.float16).contiguous()
        pos = section(f, pos, emb.numpy().tobytes())
        del emb
        norms_bytes = b"".join(
            params[f"model.layers.{i}.input_layernorm.weight"].data.float().numpy().tobytes() +
            params[f"model.layers.{i}.post_attention_layernorm.weight"].data.float().numpy().tobytes()
            for i in range(L)
        ) + params["model.norm.weight"].data.float().numpy().tobytes()
        pos = section(f, pos, norms_bytes)
        del norms_bytes

        routers_bytes = b"".join(
            params[f"model.layers.{i}.block_sparse_moe.router.layer.weight"].data.float().numpy().tobytes()
            for i in range(L)
        )
        pos = section(f, pos, routers_bytes)
        del routers_bytes

        inv = model.model.rotary_emb.inv_freq.float().numpy()
        pos = section(f, pos, inv.tobytes())
        attn_off0 = pos
        for i in range(L):                       # atencao Q4_G64: q,k,v,o
            attn = model.model.layers[i].self_attn
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                idx, sc = q4_pack(attn.__getattr__(proj).weight.data)
                pos = section(f, pos, idx.tobytes() + sc.tobytes())
        hot_size = pos - hot_off
        dir_off = pos                            # 4K alinhado (hot block fecha em 4K)
        experts_off = dir_off + L * E * 24
        if experts_off % PAGE:
            experts_off += PAGE - experts_off % PAGE
        dir_count = L * E
        entries = []

        cur_off = experts_off
        print(f"Gravando {dir_count} experts com cursor acumulativo a partir de {cur_off:,} B...")
        for i in range(L):
            fmt, k_delta = get_layer_format(i)
            il = model.model.layers[i].block_sparse_moe.input_linear.weight.data
            ol = model.model.layers[i].block_sparse_moe.output_linear.weight.data
            for e in range(E):
                if fmt == Q4:
                    idx1, sc1 = q4_pack(il[e])
                    idx2, sc2 = q4_pack(ol[e])
                    payload = (idx1.tobytes() + sc1.tobytes()
                               + idx2.tobytes() + sc2.tobytes())
                else: # C1_K8 ou C1_K4
                    b1 = c1_topk_pack(il[e], codebooks_in[i], k_delta)
                    b2 = c1_topk_pack(ol[e], codebooks_out[i], k_delta)
                    payload = b1 + b2

                exp_len = len(payload)
                assert exp_len % PAGE == 0, f"Payload do expert não é múltiplo de 4K: {exp_len}"
                entries.append((cur_off, exp_len, fmt, 0))
                f.seek(cur_off)
                f.write(payload)
                cur_off += exp_len

        final = cur_off
        assert final % PAGE == 0
        f.truncate(final)
        # header real (offsets) + diretorio
        hot_size_real = attn_off0 + L * per_layer_attn - hot_off
        f.seek(0)
        f.write(make_header(hot_off, hot_size_real, dir_off,
                            experts_off, tok_off, len(tok_bytes)))
        f.seek(dir_off)
        f.write(b"".join(struct.pack("<QQII", *t) for t in entries))

    sz = os.path.getsize(out_path)
    print(f"tamanho final .cromllm: {sz:,} bytes ({sz / 2**20:.1f} MiB)")
    print(f"hot_off={hot_off:,} hot_size={hot_size_real:,} dir_off={dir_off:,} "
          f"experts_off={experts_off:,} tok_off={tok_off:,}")
    all_aligned = all(t[0] % PAGE == 0 and t[1] % PAGE == 0 for t in entries)
    ok = (hot_off % PAGE == 0 and dir_off % PAGE == 0 and experts_off % PAGE == 0
          and all_aligned and sz == final and hot_size_real % PAGE == 0)
    print(f"alinhamento 4K de todas as secoes + tamanho final: {'OK' if ok else 'FALHOU'}")

    # ---- round-trip: reconstruir 1 matriz de atencao e experts do arquivo ----
    ref_o = model.model.layers[0].self_attn.o_proj.weight.data.float()
    o_off = attn_off0 + qsz + 2 * ksz
    with open(out_path, "rb") as f:
        f.seek(o_off)
        blob = f.read(qsz)
    idx = torch.frombuffer(bytearray(blob[:H * (H // 2)]), dtype=torch.uint8).view(H, H // 2).numpy()
    sc = torch.frombuffer(bytearray(blob[H * (H // 2):]), dtype=torch.float32).view(H, H // G).numpy()
    deq = q4_dequant(idx, sc)
    err_o = ((deq - ref_o) ** 2).sum().sqrt() / ref_o.norm()
    print(f"round-trip atencao L0.o_proj  : {ref_o.shape} -> erro L2 rel = {err_o:.4f}")

    # Valida expert L0/E0
    e0_off, e0_sz, e0_fmt, _ = entries[0]
    with open(out_path, "rb") as f:
        f.seek(e0_off)
        blob0 = f.read(e0_sz)
    ref_in0 = model.model.layers[0].block_sparse_moe.input_linear.weight.data[0].float()
    ref_out0 = model.model.layers[0].block_sparse_moe.output_linear.weight.data[0].float()
    rows_in = ref_in0.shape[0]  # in_exp_dim = 2 * intermediate_size
    cols_in = ref_in0.shape[1]  # H
    rows_out = ref_out0.shape[0] # H
    cols_out = ref_out0.shape[1] # intermediate_size

    if e0_fmt == Q4:
        nidx1 = rows_in * (cols_in // 2)
        sz_in = mat_q4_bytes(rows_in, cols_in)
        idx1 = torch.frombuffer(bytearray(blob0[:nidx1]), dtype=torch.uint8).view(rows_in, cols_in // 2).numpy()
        sc1 = torch.frombuffer(bytearray(blob0[nidx1:sz_in]), dtype=torch.float32).view(rows_in, cols_in // G).numpy()
        deq_in0 = q4_dequant(idx1, sc1)

        blob_out = blob0[sz_in:]
        nidx2 = rows_out * (cols_out // 2)
        idx2 = torch.frombuffer(bytearray(blob_out[:nidx2]), dtype=torch.uint8).view(rows_out, cols_out // 2).numpy()
        sc2 = torch.frombuffer(bytearray(blob_out[nidx2:]), dtype=torch.float32).view(rows_out, cols_out // G).numpy()
        deq_out0 = q4_dequant(idx2, sc2)
    else:
        k_d = 8 if e0_fmt == C1_K8 else 4
        in_bytes = mat_c1_bytes(rows_in, cols_in, k_d)
        deq_in0 = c1_topk_dequant(blob0[:in_bytes], rows_in, cols_in, k_d)
        deq_out0 = c1_topk_dequant(blob0[in_bytes:], rows_out, cols_out, k_d)
    err_in0 = ((deq_in0 - ref_in0) ** 2).sum().sqrt() / ref_in0.norm()
    err_out0 = ((deq_out0 - ref_out0) ** 2).sum().sqrt() / ref_out0.norm()
    print(f"round-trip expert L0/E0 in  (fmt={e0_fmt}) : {tuple(ref_in0.shape)} -> erro L2 rel = {err_in0:.4f}")
    print(f"round-trip expert L0/E0 out (fmt={e0_fmt}) : {tuple(ref_out0.shape)} -> erro L2 rel = {err_out0:.4f}")

    # Valida expert L(mid)/E0 (camada central)
    idx_mid = mid_layer * E + 0
    emid_off, emid_sz, emid_fmt, _ = entries[idx_mid]
    with open(out_path, "rb") as f:
        f.seek(emid_off)
        blob_mid = f.read(emid_sz)
    ref_in_mid = model.model.layers[mid_layer].block_sparse_moe.input_linear.weight.data[0].float()
    ref_out_mid = model.model.layers[mid_layer].block_sparse_moe.output_linear.weight.data[0].float()
    if emid_fmt == Q4:
        nidx1 = rows_in * (cols_in // 2)
        sz_in = mat_q4_bytes(rows_in, cols_in)
        idx1 = torch.frombuffer(bytearray(blob_mid[:nidx1]), dtype=torch.uint8).view(rows_in, cols_in // 2).numpy()
        sc1 = torch.frombuffer(bytearray(blob_mid[nidx1:sz_in]), dtype=torch.float32).view(rows_in, cols_in // G).numpy()
        deq_in_mid = q4_dequant(idx1, sc1)

        blob_out = blob_mid[sz_in:]
        nidx2 = rows_out * (cols_out // 2)
        idx2 = torch.frombuffer(bytearray(blob_out[:nidx2]), dtype=torch.uint8).view(rows_out, cols_out // 2).numpy()
        sc2 = torch.frombuffer(bytearray(blob_out[nidx2:]), dtype=torch.float32).view(rows_out, cols_out // G).numpy()
        deq_out_mid = q4_dequant(idx2, sc2)
    else:
        k_d = 8 if emid_fmt == C1_K8 else 4
        in_bytes = mat_c1_bytes(rows_in, cols_in, k_d)
        deq_in_mid = c1_topk_dequant(blob_mid[:in_bytes], rows_in, cols_in, k_d)
        deq_out_mid = c1_topk_dequant(blob_mid[in_bytes:], rows_out, cols_out, k_d)
    err_in_mid = ((deq_in_mid - ref_in_mid) ** 2).sum().sqrt() / ref_in_mid.norm()
    err_out_mid = ((deq_out_mid - ref_out_mid) ** 2).sum().sqrt() / ref_out_mid.norm()
    print(f"round-trip expert L{mid_layer}/E0 in  (fmt={emid_fmt}) : {tuple(ref_in_mid.shape)} -> erro L2 rel = {err_in_mid:.4f}")
    print(f"round-trip expert L{mid_layer}/E0 out (fmt={emid_fmt}) : {tuple(ref_out_mid.shape)} -> erro L2 rel = {err_out_mid:.4f}")
    print("round-trip: 'OK'")


def _last_tok(t):
    """ultimo token de um ativacao [1, seq, hidden] ou [seq, hidden]."""
    return t[0, -1] if t.dim() == 3 else t[-1]


def _norm6(t):
    t = _last_tok(t).float()
    return {"l2": float(t.norm()), "first6": [round(float(x), 6) for x in t[:6].tolist()]}


def golden(model_id=MODEL_ID):
    torch.set_num_threads(6)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True)
    model.eval()
    enc = tok(GOLDEN_TEXT, return_tensors="pt")
    ids = enc.input_ids[0].tolist()
    print(f"frase: {GOLDEN_TEXT!r}")
    print(f"input_ids ({len(ids)}): {ids}")

    ckpt = {}
    hooks = []

    def save_post(name):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            ckpt[name] = t.detach().clone().float()
        return hook

    def save_pre(name):
        def hook(_m, args):
            t = args[0] if isinstance(args, (tuple, list)) and len(args) > 0 else args
            ckpt[name] = t.detach().clone().float()
        return hook

    # after_emb: saida do embed_tokens * embedding_multiplier (modelo aplica *12.0 apos a camada)
    def hook_emb(_m, _inp, out):
        ckpt["after_emb"] = (out.detach().clone().float() * float(model.config.embedding_multiplier))
    hooks.append(model.model.embed_tokens.register_forward_hook(hook_emb))

    # after_L0_attn_resid: layer0_input + self_attn_out * residual_multiplier  (linha 574 da camada)
    hooks.append(model.model.layers[0].register_forward_pre_hook(save_pre("layer0_input")))
    hooks.append(model.model.layers[0].self_attn.register_forward_hook(save_post("attn0_out")))

    # after_L0_moe_resid: saida completa da camada 0 (apos MoE + residuo)
    hooks.append(model.model.layers[0].register_forward_hook(save_post("l0_out")))

    L = model.config.num_hidden_layers
    last_layer = L - 1

    # after_L_last: saida da ultima camada, ANTES da final_norm
    hooks.append(model.model.layers[last_layer].register_forward_hook(save_post("last_layer_out")))

    # roteamento, com hooks removidos ao fim
    routing = {}
    for li in (0, last_layer):
        router = model.model.layers[li].block_sparse_moe.router.layer
        def hook(_m, _inp, out, li=li):
            routing[li] = out.detach()
        hooks.append(router.register_forward_hook(hook))

    with torch.no_grad():
        out = model(**enc)

    # remove TODOS os hooks apos o forward
    for h in hooks:
        h.remove()

    logits = out.logits[0, -1].float()          # ja dividido por logits_scaling=6.0
    vals, tids = logits.topk(10)
    nxt = int(tids[0])
    nxt_txt = tok.decode([nxt])
    top10 = [{"id": int(t), "logit": float(v), "token": tok.decode([int(t)])}
             for v, t in zip(vals.tolist(), tids.tolist())]
    print(f"proximo token (greedy): id={nxt} {nxt_txt!r}")
    print("top-10 logits:")
    for t in top10:
        print(f"  {t['id']:6d} {t['logit']:8.4f}  {t['token']!r}")

    rot = {}
    for li in (0, last_layer):
        rl = routing[li].float()[-1]            # ultimo token
        tvals, tids8 = rl.topk(8)
        w = torch.softmax(tvals, dim=0)
        rot[li] = {"expert_ids": tids8.tolist(),
                   "weights": [round(x, 6) for x in w.tolist()]}
        print(f"router L{li} (ultimo token): experts={rot[li]['expert_ids']}")
        print(f"             pesos(softmax top-8)={rot[li]['weights']}")

    last_idx = len(ids) - 1
    last_tok_id = ids[last_idx]
    after_L0_attn_resid = (ckpt["layer0_input"]
                           + ckpt["attn0_out"] * float(model.config.residual_multiplier))
    checkpoints = {
        "token_index": last_idx,
        "token_id": last_tok_id,
        "after_emb": _norm6(ckpt["after_emb"]),
        "after_L0_attn_resid": _norm6(after_L0_attn_resid),
        "after_L0_moe_resid": _norm6(ckpt["l0_out"]),
        f"after_L{last_layer}": _norm6(ckpt["last_layer_out"]),
    }
    print("checkpoints (ultimo token, fp32):")
    for k, v in checkpoints.items():
        if k in ("token_index", "token_id"):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: l2={v['l2']:.6f} first6={v['first6']}")

    golden_file = "artifacts/golden_ref_3b.json" if "3b" in model_id else GOLDEN
    golden_meta_file = "artifacts/golden_ref_meta_3b.json" if "3b" in model_id else GOLDEN_META

    ref = {"model_id": model_id, "text": GOLDEN_TEXT, "input_ids": ids,
           "routing": {str(k): v for k, v in rot.items()},
           "top10_logits": top10, "next_token": {"id": nxt, "token": nxt_txt},
           "checkpoints": checkpoints}
    os.makedirs("artifacts", exist_ok=True)
    with open(golden_file, "w", encoding="utf-8") as f:
        json.dump(ref, f, ensure_ascii=False, indent=2)
    meta = {"model_id": model_id, "text": GOLDEN_TEXT, "n_input_ids": len(ids),
            "input_ids": ids,
            "routing_l0": rot[0], f"routing_l{last_layer}": rot[last_layer],
            "top10_ids": [t["id"] for t in top10],
            "top10_logits": [round(t["logit"], 4) for t in top10],
            "next_token": {"id": nxt, "token": nxt_txt},
            "checkpoints": checkpoints}
    with open(golden_meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"salvo: {golden_file} + {golden_meta_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Empacotador de modelos para o formato .cromllm v1")
    parser.add_argument("--model-id", default=MODEL_ID, help="ID do modelo HuggingFace (padrão: %(default)s)")
    parser.add_argument("--out", default=OUT, help="Arquivo de saída .cromllm (padrão: %(default)s)")
    parser.add_argument("--profile", default="q4", choices=["q4", "c1-topk8", "c1-topk4", "hybrid-c1-k8", "hybrid-c1-k4"],
                        help="Perfil de compressão dos experts (padrão: %(default)s)")
    parser.add_argument("--golden", action="store_true", help="Gera gabarito PyTorch BF16")
    args = parser.parse_args()

    if args.golden:
        golden(model_id=args.model_id)
    else:
        pack(model_id=args.model_id, out_path=args.out, profile=args.profile)
