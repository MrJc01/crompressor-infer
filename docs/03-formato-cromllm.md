# Formato binário `.cromllm` — especificação v1

**Projeto:** crompressor-infer · **Data:** 2026-11-14 · **Status:** implementado no
`tools/pack_cromllm.py` (granite-3.1-1b-a400m-instruct) · **Alinhamento:** 4096 bytes
(por `O_DIRECT` — medido no EXP-001)

## 1. Princípios

1. **Alinhamento a 4096 B em toda a fronteira de seção** — toda seção começa em
   múltiplo de 4096 e os payloads "quentes" de leitura única (experts) têm tamanho
   múltiplo de 4096, permitindo `pread()` único por expert com `O_DIRECT`.
2. **Blocos Hot vs Cold:** o bloco Hot (embeddings + norms + routers + atenção +
   tokenizador) fica residente em RAM; os experts (Cold) são a seção que o motor
   pode manter em RAM (Q4) ou paginar do SSD (Caminho A, cache LRU do EXP-005).
3. **Formato de peso único no v1:** `Q4_G64` (id = 1), exatamente o layout
   validado pelo kernel `kv_q4` do EXP-005 (SIMD↔escalar ≤ 2e-07).
4. **Tokenizer embutido:** `tokenizer.json` completo embutido na seção de
   tokenizador — o motor Go lê o vocabulário direto do arquivo.

## 2. Cabeçalho fixo (4096 bytes, little-endian)

| Campo | Tipo | Valor v1 / Significado |
|---|---|---|
| `magic` | char[8] | `"CROMLM01"` |
| `version` | u32 | 1 |
| `vocab_size` | u32 | 49155 |
| `hidden_size` | u32 | 1024 |
| `intermediate_size` | u32 | 512 |
| `num_layers` | u32 | 24 |
| `num_attn_heads` | u32 | 16 |
| `num_kv_heads` | u32 | 8 |
| `num_local_experts` | u32 | 32 |
| `num_experts_per_tok` | u32 | 8 |
| `head_dim` | u32 | 64 |
| `rope_theta` | f64 | 1 500 000.0 |
| `rms_norm_eps` | f32 | 1e-06 |
| `embedding_multiplier` | f32 | 12.0 |
| `attention_multiplier` | f32 | 0.015625 (= 1/√head_dim) |
| `residual_multiplier` | f32 | 0.22 |
| `logits_scaling` | f32 | 6.0 |
| `embed_format` | u32 | 0 = F16 (usado no pack atual) |
| `attn_format` | u32 | 1 = Q4_G64 |
| `expert_format` | u32 | 1 = Q4_G64 |
| `group_size` | u32 | 64 |
| `hot_offset` | u64 | offset absoluto do bloco Hot |
| `hot_size` | u64 | tamanho do bloco Hot (sem padding final) |
| `expert_dir_offset` | u64 | offset do diretório de experts |
| `expert_dir_count` | u32 | 768 (24×32) |
| `expert_dir_entry_size` | u32 | 24 bytes |
| `experts_offset` | u64 | offset da seção de experts |
| `tokenizer_offset` | u64 | offset do `tokenizer.json` embutido |
| `tokenizer_size` | u64 | tamanho exato do JSON (sem padding) |
| `reserved` | — | resto do header zerado |

## 3. Bloco Hot (ordem fixa, cada sub-seção alinhada a 4 KiB)

| Sub-seção | Conteúdo | Formato | Tamanho (granite-1b) |
|---|---|---|---|
| `embed_tokens` | embedding (pesos do lm_head **tied** — reuse para o logits) | F16 | 49 155×1024×2 = 100 661 760 B |
| `norms` | 24× `input_layernorm` + 24× `post_attention_layernorm` + `final norm` (nesta ordem: L0.in, L0.post, L1.in, …, L23.post, final) | FP32 | 49×1024×4 = 200 704 B |
| `routers` | `router.layer.weight` de cada camada L0..L23, linha a linha | FP32 | 24×32×1024×4 = 3 145 728 B |
| `rotary_inv_freq` | `rotary_emb.inv_freq` (32 valores = head_dim/2) | FP32 | 128 B |
| `attn` | por camada L0..L23, nesta ordem: `q_proj`, `k_proj`, `v_proj`, `o_proj` | Q4_G64 | 1 769 472 B/camada → 42 467 328 B |

Notas de atenção: `k_proj`/`v_proj` têm 512 linhas (8 KV heads × 64); RoPE é
**split-half** (`rotate_half`: `cat(-x2, x1)`); o multiplicador
`attention_multiplier` aplica-se PÓS-`q@k.T` (não em q); softmax em fp32.

## 4. Tokenizador

`tokenizer.json` completo (offset/size no header), seguido de padding a 4 KiB.

## 5. Diretório de Experts

Array de `num_layers × num_local_experts` (768) entradas de **24 bytes**,
ordenadas por camada e depois por expert (`layer*32 + expert`):

```c
struct expert_dir_entry {
    uint64_t offset_bytes;  // offset absoluto no .cromllm (múltiplo de 4096)
    uint64_t size_bytes;    // tamanho do payload (múltiplo de 4096)
    uint32_t format_id;     // 1 = Q4_G64 (permite perfis híbridos por expert)
    uint32_t reserved;
};
```

Cada expert é gravado **contíguo e alinhado a 4 KiB**: payload = `input_linear` +
`output_linear` da MESMA expert, cabendo em **1 único `pread()`**.

## 6. Payload de expert (Q4_G64)

Para cada expert, em ordem: `input_linear` `[32 experts, 1024 out, 1024 in]` →
matriz `[1024, 1024]` do expert; depois `output_linear` `[32, 1024 out, 512 in]`
→ matriz `[1024, 512]` do expert. Ambas em Q4_G64:

- Grupos de **64 pesos ao longo da dimensão de entrada (cols)**.
- Vetor `idx` contíguo por matriz: por linha, `cols/2` bytes; byte `c` do grupo
  `g` = `(q[c] + 8) | ((q[c+32] + 8) << 4)`, com `c ∈ [0,32)`,
  `q[c]` = peso `g*64 + c`, `q[c+32]` = peso `g*64 + c + 32` (nibble low =
  peso par-relativo, high = peso +32).
- Vetor `scales` contíguo por matriz: `cols/64` **float32** por linha
  (1 escala por grupo de 64).
- Valores: `q ∈ [-7, 7]` int; reconstrução `w = q * scale`.

**Tamanho exato por expert (granite-1b):**
`input_linear` [1024, 1024]: idx 1024×512 = 524 288 B + escalas 1024×16×4 =
65 536 B → **589 824 B**; `output_linear` [1024, 512]: idx 1024×256 = 262 144 B +
escalas 1024×8×4 = 32 768 B → **294 912 B**; **total 884 736 B por expert**
(= 216 × 4096, sem padding extra!). 768 experts × 884 736 B =
**679 477 248 B (648,2 MB)**.

## 7. Matemática de referência (must-ler do motor — EXP-007)

- `x_emb * 12.0` (embedding_multiplier); RMSNorm com `eps=1e-06`.
- Resíduos: `h = residual + h * 0.22` (atenção **e** MoE).
- RoPE split-half, `theta=1 500 000`; `attention_multiplier=0.015625`
  aplicado em `q@k.T` (equivale a 1/√64).
- GQA: 16 Q-heads, 8 KV-heads (repeat_kv ×2).
- **Router: `topk(8)` ANTES do `softmax`** — softmax só nos 8 logits
  selecionados (renormaliza os pesos dos 8 experts para somar 1).
- MoE: `input_linear [gate, up]` — `chunk(2)` em `-1`: **primeira metade =
  gate (silu), segunda = up**; `silu(gate) * up`; depois `output_linear`;
  saída multiplicada pelos pesos renormalizados e somada por índice.
- Expert matmuls: `x @ W.T` (pesos armazenados `[out, in]` — F.linear).
- Logits finais: `lm_head(embed) / 6.0` (logits_scaling).

## 8. Uso do diretório no motor (Caminho A)

`pread(fd, buf, size_bytes, offset_bytes)` com `O_DIRECT` baixa 1 expert
inteiro (884 736 B) em ~2,6 ms no SSD SATA do EXP-001 (344,5 MB/s em 1 MiB).
Com cache LRU cap 24/32 → ~21 MB/token (EXP-005) ≈ 60 ms/token de I/O puro;
com experts residentes (Q4, 648 MB) o disco não entra.
