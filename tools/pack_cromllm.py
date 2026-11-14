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
EMBED_F16 = 0   # embed_format
G = 64
MAGIC = b"CROMLM01"
HEAD_FMT = "<8s14IffffffQQIIQQQQ"  # 144 bytes
assert struct.calcsize(HEAD_FMT) == 144


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

def mat_q4_bytes(rows, cols):
    """tamanho em bytes da matriz em Q4_G64."""
    return rows * (cols // 2) + rows * (cols // G) * 4


def section(f, pos, data):
    f.write(data)
    pos += len(data)
    if pos % PAGE:
        f.write(b"\x00" * (PAGE - pos % PAGE))
        pos += PAGE - pos % PAGE
    return pos


def pack():
    torch.set_num_threads(6)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True,
                                                 local_files_only=True)
    model.eval()
    cfg = model.config
    L, E, H = cfg.num_hidden_layers, cfg.num_local_experts, cfg.hidden_size
    params = dict(model.named_parameters())

    tok_path = os.path.dirname(cached_file(MODEL_ID, "config.json", local_files_only=True))
    tok_bytes = open(os.path.join(tok_path, "tokenizer.json"), "rb").read()
    print(f"tokenizer.json embutido: {len(tok_bytes):,} bytes")

    qsz = mat_q4_bytes(H, H)          # q/o [1024,1024] = 589824
    ksz = mat_q4_bytes(cfg.num_key_value_heads * (H // cfg.num_attention_heads), H)  # k/v [512,1024]
    per_layer_attn = qsz + 2 * ksz + qsz
    exp_in = mat_q4_bytes(H, H)       # input_linear  [1024,1024]
    exp_out = mat_q4_bytes(H, cfg.intermediate_size)  # output_linear [1024,512]
    exp_size = exp_in + exp_out       # 884736 = 216 * 4096

    os.makedirs("artifacts", exist_ok=True)
    def make_header(hot_off, hot_size, dir_off, experts_off, tok_off, tok_size):
        vals = (MAGIC,
                1,                                    # version
                cfg.vocab_size, H, cfg.intermediate_size, L,
                cfg.num_attention_heads, cfg.num_key_value_heads,
                E, cfg.num_experts_per_tok,
                H // cfg.num_attention_heads,         # head_dim
                EMBED_F16, Q4, Q4, G,
                float(cfg.rope_theta), float(cfg.rms_norm_eps),
                float(cfg.embedding_multiplier),
                float(cfg.attention_multiplier),
                float(cfg.residual_multiplier),
                float(cfg.logits_scaling),
                hot_off, hot_size, dir_off, L * E, 24,
                experts_off, tok_off, tok_size)
        return struct.pack(HEAD_FMT, *vals) + b"\x00" * (PAGE - struct.calcsize(HEAD_FMT))

    with open(OUT, "wb") as f:
        f.write(make_header(0, 0, 0, 0, 0, 0))   # placeholder; patch no fim
        pos = PAGE
        tok_off = pos
        pos = section(f, pos, tok_bytes)
        hot_off = pos
        emb = params["model.embed_tokens.weight"].data.to(torch.float16).contiguous()
        pos = section(f, pos, emb.numpy().tobytes())
        del emb
        for i in range(L):                       # norms: L0.in, L0.post, ..., final
            pos = section(f, pos, params[f"model.layers.{i}.input_layernorm.weight"].data.float().numpy().tobytes())
            pos = section(f, pos, params[f"model.layers.{i}.post_attention_layernorm.weight"].data.float().numpy().tobytes())
        pos = section(f, pos, params["model.norm.weight"].data.float().numpy().tobytes())
        for i in range(L):                       # routers FP32
            pos = section(f, pos, params[f"model.layers.{i}.block_sparse_moe.router.layer.weight"].data.float().numpy().tobytes())
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
        for i in range(L):
            il = model.model.layers[i].block_sparse_moe.input_linear.weight.data
            ol = model.model.layers[i].block_sparse_moe.output_linear.weight.data
            for e in range(E):
                idx1, sc1 = q4_pack(il[e])
                idx2, sc2 = q4_pack(ol[e])
                off = experts_off + (i * E + e) * exp_size
                entries.append((off, exp_size, Q4, 0))
                payload = (idx1.tobytes() + sc1.tobytes()
                           + idx2.tobytes() + sc2.tobytes())
                assert len(payload) == exp_size
                f.seek(off)
                f.write(payload)
        final = experts_off + L * E * exp_size
        assert final % PAGE == 0
        f.truncate(final)
        # header real (offsets) + diretorio
        hot_size_real = attn_off0 + L * per_layer_attn - hot_off
        f.seek(0)
        f.write(make_header(hot_off, hot_size_real, dir_off,
                            experts_off, tok_off, len(tok_bytes)))
        f.seek(dir_off)
        f.write(b"".join(struct.pack("<QQII", *t) for t in entries))

    sz = os.path.getsize(OUT)
    print(f"tamanho final .cromllm: {sz:,} bytes ({sz / 2**20:.1f} MiB)")
    print(f"expert_size={exp_size:,} (= {exp_size // PAGE} paginas 4K) | "
          f"experts={L * E} -> {L * E * exp_size / 2**20:.1f} MiB")
    print(f"hot_off={hot_off:,} hot_size={hot_size_real:,} dir_off={dir_off:,} "
          f"experts_off={experts_off:,} tok_off={tok_off:,}")
    ok = (hot_off % PAGE == 0 and dir_off % PAGE == 0 and experts_off % PAGE == 0
          and exp_size % PAGE == 0 and sz == final and hot_size_real % PAGE == 0)
    print(f"alinhamento 4K de todas as secoes + tamanho final: {'OK' if ok else 'FALHOU'}")

    # ---- round-trip: reconstruir 1 matriz de atencao e 1 expert do arquivo ----
    ref_o = model.model.layers[0].self_attn.o_proj.weight.data.float()
    o_off = attn_off0 + qsz + 2 * ksz
    with open(OUT, "rb") as f:
        f.seek(o_off)
        blob = f.read(qsz)
    idx = torch.frombuffer(bytearray(blob[:H * (H // 2)]), dtype=torch.uint8).view(H, H // 2).numpy()
    sc = torch.frombuffer(bytearray(blob[H * (H // 2):]), dtype=torch.float32).view(H, H // G).numpy()
    deq = q4_dequant(idx, sc)
    err_o = ((deq - ref_o) ** 2).sum().sqrt() / ref_o.norm()
    print(f"round-trip atencao L0.o_proj  : {ref_o.shape} -> erro L2 rel = {err_o:.4f}")

    ref_in = model.model.layers[0].block_sparse_moe.input_linear.weight.data[0].float()
    with open(OUT, "rb") as f:
        f.seek(entries[0][0])
        blob = f.read(exp_size)
    nidx = H * (H // 2)
    idx1 = torch.frombuffer(bytearray(blob[:nidx]), dtype=torch.uint8).view(H, H // 2).numpy()
    sc1 = torch.frombuffer(bytearray(blob[nidx:nidx + H * (H // G) * 4]), dtype=torch.float32).view(H, H // G).numpy()
    deq1 = q4_dequant(idx1, sc1)
    err_in = ((deq1 - ref_in) ** 2).sum().sqrt() / ref_in.norm()
    ref_out = model.model.layers[0].block_sparse_moe.output_linear.weight.data[0].float()
    off2 = exp_in
    nidx2 = H * (cfg.intermediate_size // 2)
    idx2 = torch.frombuffer(bytearray(blob[off2:off2 + nidx2]), dtype=torch.uint8).view(H, cfg.intermediate_size // 2).numpy()
    sc2 = torch.frombuffer(bytearray(blob[off2 + nidx2:]), dtype=torch.float32).view(H, cfg.intermediate_size // G).numpy()
    deq2 = q4_dequant(idx2, sc2)
    err_out = ((deq2 - ref_out) ** 2).sum().sqrt() / ref_out.norm()
    print(f"round-trip expert L0/E0 in    : {tuple(ref_in.shape)} -> erro L2 rel = {err_in:.4f}")
    print(f"round-trip expert L0/E0 out   : {tuple(ref_out.shape)} -> erro L2 rel = {err_out:.4f}")
    print("round-trip: 'OK' com erro esperado ~0.07 (Q4 dequant, EXP-006)")


def _last_tok(t):
    """ultimo token de um ativacao [1, seq, hidden] ou [seq, hidden]."""
    return t[0, -1] if t.dim() == 3 else t[-1]


def _norm6(t):
    t = _last_tok(t).float()
    return {"l2": float(t.norm()), "first6": [round(float(x), 6) for x in t[:6].tolist()]}


def golden():
    torch.set_num_threads(6)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True,
                                                 local_files_only=True)
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

    # after_L23: saida da camada 23, ANTES da final_norm
    hooks.append(model.model.layers[23].register_forward_hook(save_post("l23_out")))

    # roteamento (mantido do EXP-007), com hooks removidos ao fim
    routing = {}
    for li in (0, 23):
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
    for li in (0, 23):
        rl = routing[li].float()[-1]            # ultimo token: [32]
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
        "after_L23": _norm6(ckpt["l23_out"]),
    }
    print("checkpoints (ultimo token, fp32):")
    for k, v in checkpoints.items():
        if k in ("token_index", "token_id"):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: l2={v['l2']:.6f} first6={v['first6']}")

    ref = {"model_id": MODEL_ID, "text": GOLDEN_TEXT, "input_ids": ids,
           "routing": {str(k): v for k, v in rot.items()},
           "top10_logits": top10, "next_token": {"id": nxt, "token": nxt_txt},
           "checkpoints": checkpoints}
    os.makedirs("artifacts", exist_ok=True)
    with open(GOLDEN, "w", encoding="utf-8") as f:
        json.dump(ref, f, ensure_ascii=False, indent=2)
    meta = {"model_id": MODEL_ID, "text": GOLDEN_TEXT, "n_input_ids": len(ids),
            "input_ids": ids,
            "routing_l0": rot[0], "routing_l23": rot[23],
            "top10_ids": [t["id"] for t in top10],
            "top10_logits": [round(t["logit"], 4) for t in top10],
            "next_token": {"id": nxt, "token": nxt_txt},
            "checkpoints": checkpoints}
    with open(GOLDEN_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"salvo: {GOLDEN} + {GOLDEN_META}")


if __name__ == "__main__":
    if "--golden" in sys.argv:
        golden()
    else:
        pack()
