# Crompressor-Infer — Próximos Passos (Roadmap EXP-011+)

> Documento de retomada. Consolida o que já foi **validado até o EXP-010** e os
> marcos planejados para as próximas sessões. Sempre comparar qualquer novo
> experimento contra o baseline registrado em `docs/01-ambiente-e-baseline.md`.

## 1. Estado validado até o EXP-010

### Motor e numérica

- **Motor C + AVX2/FMA/F16C** (`engine/crom_infer.c`) operando sobre o formato
  `.cromllm` alinhado a 4 KiB, com streaming `O_DIRECT`, LRU de experts e
  KV-cache.
- **Exatidão vs PyTorch BF16**: checkpoints de camada (`after_emb`,
  `after_L0_attn_resid`, `after_L0_moe_resid`, `after_L23`), roteamento top-8 e
  `next_token_id` validados (EXP-008 / EXP-009 / EXP-009B).
- **`kv_q4` AVX2+FMA** no `q4_matvec` e FMA no `lm_head` (EXP-009B): decode no
  modo RAM saltou de ~15,5 → **~57 tok/s** com exatidão preservada.

### CLI Go+CGO (EXP-010)

- `bin/crom-infer` (Go 1.22 + CGO) estável em **~59 tok/s** (RAM 768).
  Os dois bugs da integração foram corrigidos:
  - **`runtime.LockOSThread()`** — sem ele a goroutine migra entre OS threads e
    o libgomp reconstrói/contende o time OpenMP a cada `crom_forward` (~100×
    mais lento, com travamentos).
  - **`debug.SetGCPercent(-1)`** — evita que workers do GC roubem núcleos do
    time OpenMP durante a inferência.
  - Timers `attn_ms` / `moe_ms` / `lm_head_ms` agora vêm de `crom_stats()`.

### Tokenizador nativo (EXP-010)

- `internal/tokenizer/tokenizer.go` reproduz o `GPT2TokenizerFast` do Granite:
  split de `added_tokens` antes do BPE, tabela byte↔unicode GPT-2,
  pré-tokenização ByteLevel com lookahead `\s+(?!\S)`, merges por rank e chat
  template fiel.
- **100 % `ids_go == ids_hf`** nos 5 casos de `tools/check_tokenizer.py`
  (prompt cru + 4 prompts com chat template).
- `eos_token_id` real = `0` (`<|end_of_text|>`).

### Baseline EXP-010 (4 prompts, 6 threads)

| Engine | decode tok/s | prefill tok/s | RSS MB | cache hit % | ssd_stream MB |
|---|---|---|---|---|---|
| PyTorch BF16 | 3,81 | — | 3.482 (pico) | — | — |
| Ollama Q8_0 | 41,32 | 587,82 | ~1.740 | — | — |
| **crom-infer RAM (768)** | **47,78** | 52,92 | **871** | 100,00 | 0,00 |
| **crom-infer SSD (576)** | **15,06** | 8,93 | **709** | 90,89 | 2.971,90 |

Tokens do modo SSD (576) **bit-idênticos** ao modo RAM (768) nos 4 prompts.

### Gargalos conhecidos

- **Prefill**: ~53 tok/s contra ~588 tok/s do Ollama — falta processamento em
  lote (hoje é 1 token por `crom_forward`).
- **Streaming SSD**: domina o tempo no modo 576 (`moe_ms` ≈ 15,9 s vs 2,3 s no
  modo RAM); a espera de I/O não é sobreposta ao cálculo.
- **Contexto**: `CROM_MAX_SEQ = 512` limita o histórico.

---

## 2. Marco Futuro A — Modelo maior que a RAM (Território Colibrì) — CONCLUÍDO (EXP-015)

**Objetivo:** provar que o motor continua útil quando o modelo **não cabe** na
RAM, que é o diferencial real frente a soluções `mmap`.

- [x] **Suporte e Empacotamento de MoE 3.3B (`granite-3.1-3b-a800m-instruct`):**
  - 32 camadas, 40 experts por camada (1280 experts no total), top-8 ativos, hidden=1536, intermediate=512.
  - Empacotado em `Q4_G64` (1,84 GiB) e `Híbrido C1_TOPK_K8` (1,58 GiB).
- [x] **Duelo e Validação contra Ollama (`granite3.1-moe:3b`, Q4_K_M):**
  - Avaliação de desempenho nos 4 prompts padrão em 100% RAM e 50% RAM (640 experts no cache).
  - Economia de 22,1% no tráfego de leitura de SSD no modo híbrido vs Q4 (~8,34 GB economizados nos 4 prompts).
  - Resiliência comprovada sob restrição de memória: o `crom-infer` opera estavelmente sob limites estritos de cgroup/RAM onde processos monolíticos do Ollama sofrem terminação imediata (OOM killer).

---

## 3. Marco Futuro B — Streaming SSD mais rápido + Compressão `C1_v2` + Prefill GEMM — CONCLUÍDO (EXP-012, EXP-013)

### B.1 Prefetch assíncrono de experts — [x] CONCLUÍDO (EXP-012)
- Pipeline I/O ↔ compute com `pthread` e pinning LRU de experts.

### B.2 Suporte a `format_id = 2` (`C1_TOPK_K8`, 3,50 bits/peso) — [x] CONCLUÍDO (EXP-013A / EXP-013B)
- Codebook `d=4, K=256` normalizado por grupo de 64 + Top-K deltas esparsos com decodificação direta em AVX2.

### B.3 Batch Prefill (GEMM) — [x] CONCLUÍDO (EXP-012)
- GEMM otimizado com empacotamento contíguo de tokens para prefill de contexto.

---

## 4. Marco C — Diferencial `crompressor` (Deltas P2P e Multi-Fine-Tune) — CONCLUÍDO (EXP-014)

### C.1 `W_base + Δ` (múltiplos fine-tunes na mesma RAM) — [x] CONCLUÍDO
- Suporte a modelo Base (`granite-1b-base.cromllm`) + deltas de 1 bit por grupo de 64 (`granite-1b-instruct.cromdelta`).
- Cálculo direto no motor C (`delta_matvec_add` AVX2 unrolled com FMA) sem materializar $W_{\text{instruct}}$.
- Economia de 75,4% no armazenamento de deltas com qualidade de PPL (61.20) superior a Q4.

### C.2 Distribuição/sincronização de deltas via Rede — [x] CONCLUÍDO
- Subcomandos `serve-delta` e `pull-delta` em Go nativo (`net/http` + `crypto/sha256`) com streaming por chunks de 1 MiB e validação criptográfica ponta a ponta.

---

## 5. Higiene e Empacotamento

- [x] **Aumentar `CROM_MAX_SEQ`** e tornar contexto dinâmico no cabeçalho / flag `--ctx 2048` (**concluído no EXP-011**).
- [x] **Adicionar flag `--repeat-penalty`** na amostragem (**concluído no EXP-011**).
- [x] **Motor C e Go 100% dinâmico** por header `.cromllm` sem hardcoding de dimensões (**concluído no EXP-011**).
- [x] **Batch Prefill GEMM e Prefetch Assíncrono via pthread** (**concluído no EXP-012**).
- [x] **Layout Compacto C1_TOPK, format_id = 2 e Perfil Híbrido** (**concluído no EXP-013A**).
- [x] **Kernel C C1_TOPK direto e Duelo Híbrido vs Q4 em SSD Streaming** (**concluído no EXP-013B**).
- [x] **Multi-Fine-Tune Base + Delta 1-bit e Sync de Deltas por Rede** (**concluído no EXP-014**).
- [x] **MoE de 3.3B (Granite-3.1-3B) em Q4 e Híbrido C1_K8 vs Ollama** (**concluído no EXP-015**).
- [ ] Suporte a modelos com MoE de 64+ experts (`OLMoE-1B-7B`) sob orçamentos ultra-restritos (< 1 GB de RAM).
- [ ] Suportar **múltiplos modelos** no cabeçalho `.cromllm` (catálogo/versão).
- [x] Manter `crompressor/`, `artifacts/`, `bin/`, `*.a`, `*.o` fora do versionamento.

---

## 6. Ordem sugerida de retomada

1. **EXP-011** (Motor Dinâmico + `--ctx 2048` + `--repeat-penalty`) — **CONCLUÍDO**.
2. **EXP-012** (Batch Prefill GEMM + Prefetch Assíncrono SSD via `pthread` + pinning LRU) — **CONCLUÍDO**.
3. **EXP-013A / EXP-013B** (`C1_TOPK_K8` híbrido: 3,75 bits/peso, +19,5% tok/s em SSD 576) — **CONCLUÍDO**.
4. **EXP-014** (Multi-Fine-Tune Base+Delta 1-bit + Sync de Delta via Rede HTTP) — **CONCLUÍDO**.
5. **EXP-015** (Granite-3.1-3B MoE em Q4 e Híbrido C1_K8 vs Ollama sob pressão de RAM) — **CONCLUÍDO**.
6. **EXP-016** (Escala para MoE 7B com 64 experts — OLMoE-1B-7B — ou compressão C1_TOPK_K4 a 3.0 bits/peso).

