#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include "crom_infer.h"

static int parse_ids(const char *s, int *out, int maxn) {
    int n = 0;
    char *buf = strdup(s);
    char *tok = strtok(buf, ", ");
    while (tok && n < maxn) {
        out[n++] = atoi(tok);
        tok = strtok(NULL, ", ");
    }
    free(buf);
    return n;
}

static void vec_l2_first6(const float *v, int n, double *l2, float first6[6]) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)v[i] * (double)v[i];
    *l2 = sqrt(s);
    for (int i = 0; i < 6; i++) first6[i] = (i < n) ? v[i] : 0.0f;
}

/* top-k eficiente O(vocab*k) */
static void topk(const float *logits, int vocab, int k, int *ids_out, float *vals_out) {
    float *vals = malloc((size_t)k * sizeof(float));
    int *ids = malloc((size_t)k * sizeof(int));
    for (int i = 0; i < k; i++) { vals[i] = -INFINITY; ids[i] = -1; }
    for (int v = 0; v < vocab; v++) {
        float lv = logits[v];
        /* acha posicao de insercao (menor entre os top-k) */
        int pos = -1;
        for (int i = 0; i < k; i++) {
            if (lv > vals[i]) { pos = i; break; }
        }
        if (pos < 0) continue;
        /* desloca */
        for (int i = k - 1; i > pos; i--) { vals[i] = vals[i - 1]; ids[i] = ids[i - 1]; }
        vals[pos] = lv; ids[pos] = v;
    }
    for (int i = 0; i < k; i++) { ids_out[i] = ids[i]; if (vals_out) vals_out[i] = vals[i]; }
    free(vals);
    free(ids);
}

int main(int argc, char **argv) {
    const char *model_path = "artifacts/granite-1b.cromllm";
    int max_seq = 2048;
    int *ids = NULL;
    int n_ids = 0;
    int steps = 32;
    int threads = 6;
    int cache_experts = -1;
    int dump_ckpt = 0;
    int sync_io = 0;
    const char *ids_str = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_str = argv[++i];
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cache-experts") && i + 1 < argc) cache_experts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) max_seq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump-ckpt")) dump_ckpt = 1;
        else if (!strcmp(argv[i], "--sync-io")) sync_io = 1;
        else { fprintf(stderr, "arg desconhecido: %s\n", argv[i]); return 2; }
    }
    if (!ids_str) { fprintf(stderr, "use --ids\n"); return 2; }

    ids = malloc((size_t)max_seq * sizeof(int));
    if (!ids) { perror("malloc ids"); return 1; }
    n_ids = parse_ids(ids_str, ids, max_seq);
    if (n_ids == 0) { fprintf(stderr, "nenhum id valido em --ids\n"); free(ids); return 2; }

    omp_set_num_threads(threads);

    crom_model_t m;
    double t0 = omp_get_wtime();
    if (crom_model_open_ctx(model_path, &m, cache_experts, threads, max_seq) != 0) {
        fprintf(stderr, "falha ao abrir modelo\n"); free(ids); return 1;
    }
    crom_set_sync_io(&m, sync_io);
    double t1 = omp_get_wtime();
    double load_ms = (t1 - t0) * 1000.0;

    int H = m.hidden_size;
    int V = m.vocab_size;
    int L = m.num_layers;
    size_t logits_sz = ((size_t)V * sizeof(float) + CROM_PAGE - 1) & ~(CROM_PAGE - 1);
    float *hidden = aligned_alloc(4096, H * sizeof(float));
    float *hidden2 = aligned_alloc(4096, H * sizeof(float));
    float *logits = aligned_alloc(4096, logits_sz);
    int top10_ids[10];
    float top10_vals[10];

    int *cur = malloc((size_t)m.max_seq * sizeof(int));
    int *gen_ids = malloc((size_t)m.max_seq * sizeof(int));
    if (!cur || !gen_ids) { perror("malloc cur/gen_ids"); return 1; }
    int ncur = n_ids;
    memcpy(cur, ids, n_ids * sizeof(int));

    int n_gen = 0;
    double prefill_t0 = omp_get_wtime();
    if (dump_ckpt) {
        for (int p = 0; p < n_ids - 1; p++) {
            crom_forward(&m, cur, ncur, p, hidden, logits);
        }
    } else {
        if (n_ids - 1 > 0) {
            crom_forward_batch(&m, cur, n_ids - 1, 0, logits);
        }
    }
    double prefill_t1 = omp_get_wtime();

    if (dump_ckpt) {
        int last = ids[n_ids - 1];
        double l2; float f6[6];

        crom_embed(&m, last, hidden);
        vec_l2_first6(hidden, H, &l2, f6);
        printf("after_emb: l2=%.6f first6=[%.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
               l2, f6[0], f6[1], f6[2], f6[3], f6[4], f6[5]);

        crom_layer_attn_resid(&m, 0, hidden, n_ids - 1, hidden2);
        vec_l2_first6(hidden2, H, &l2, f6);
        printf("after_L0_attn_resid: l2=%.6f first6=[%.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
               l2, f6[0], f6[1], f6[2], f6[3], f6[4], f6[5]);
        memcpy(hidden, hidden2, H * sizeof(float));

        int top8_l0[8]; float w8_l0[8];
        crom_layer_moe_resid(&m, 0, hidden, n_ids - 1, hidden2, top8_l0, w8_l0);
        vec_l2_first6(hidden2, H, &l2, f6);
        printf("after_L0_moe_resid: l2=%.6f first6=[%.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
               l2, f6[0], f6[1], f6[2], f6[3], f6[4], f6[5]);
        printf("router L0: experts=[");
        for (int i = 0; i < 8; i++) printf("%d%s", top8_l0[i], i < 7 ? ", " : "");
        printf("] pesos=[");
        for (int i = 0; i < 8; i++) printf("%.6f%s", w8_l0[i], i < 7 ? ", " : "");
        printf("]\n");
        memcpy(hidden, hidden2, H * sizeof(float));

        for (int li = 1; li < L - 1; li++) {
            crom_layer_attn_resid(&m, li, hidden, n_ids - 1, hidden2);
            memcpy(hidden, hidden2, H * sizeof(float));
            crom_layer_moe_resid(&m, li, hidden, n_ids - 1, hidden2, NULL, NULL);
            memcpy(hidden, hidden2, H * sizeof(float));
        }

        crom_layer_attn_resid(&m, L - 1, hidden, n_ids - 1, hidden2);
        memcpy(hidden, hidden2, H * sizeof(float));
        int top8_lL[8]; float w8_lL[8];
        crom_layer_moe_resid(&m, L - 1, hidden, n_ids - 1, hidden2, top8_lL, w8_lL);
        vec_l2_first6(hidden2, H, &l2, f6);
        printf("after_L%d: l2=%.6f first6=[%.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
               L - 1, l2, f6[0], f6[1], f6[2], f6[3], f6[4], f6[5]);
        printf("router L%d: experts=[", L - 1);
        for (int i = 0; i < 8; i++) printf("%d%s", top8_lL[i], i < 7 ? ", " : "");
        printf("] pesos=[");
        for (int i = 0; i < 8; i++) printf("%.6f%s", w8_lL[i], i < 7 ? ", " : "");
        printf("]\n");

        /* lm_head sobre hidden apos ultima camada (final_norm sobre hidden2) */
        float *fn = aligned_alloc(4096, H * sizeof(float));
        float *final_norm_w = m.norms_f32 + (size_t)(m.num_layers * 2) * H;
        crom_rms_norm(final_norm_w, hidden2, H, m.rms_norm_eps, fn);
        const uint16_t *embed = m.embed_f16;
#pragma omp parallel for schedule(static)
        for (int v = 0; v < V; v++) {
            float s = 0.0f;
            const uint16_t *ev = embed + (size_t)v * H;
            int i = 0;
            for (; i + 7 < H; i += 8) {
                __m128i f16 = _mm_loadu_si128((const __m128i *)(ev + i));
                __m256 e = _mm256_cvtph_ps(f16);
                __m256 hh = _mm256_loadu_ps(fn + i);
                __m256 p = _mm256_mul_ps(e, hh);
                const float *pf = (const float *)&p;
                s += pf[0] + pf[1] + pf[2] + pf[3] + pf[4] + pf[5] + pf[6] + pf[7];
            }
            for (; i < H; i++) s += _cvtsh_ss(ev[i]) * fn[i];
            logits[v] = s / m.logits_scaling;
        }
        free(fn);

        topk(logits, V, 10, top10_ids, top10_vals);
        printf("top-10 logits:");
        for (int i = 0; i < 10; i++)
            printf(" [%d]=%.4f", top10_ids[i], top10_vals[i]);
        printf("\nnext_token_id=%d\n", top10_ids[0]);
    }

    /* geracao */
    double decode_t0 = omp_get_wtime();
    int pos = n_ids - 1;
    for (int s = 0; s < steps && pos < m.max_seq - 1; s++) {
        crom_forward(&m, cur, ncur, pos, hidden, logits);
        int nt_id; float nt_val;
        topk(logits, V, 1, &nt_id, &nt_val);
        gen_ids[n_gen++] = nt_id;
        cur[ncur++] = nt_id;
        pos++;
    }
    double decode_t1 = omp_get_wtime();

    printf("gerados %d tokens:", n_gen);
    for (int i = 0; i < n_gen; i++) printf(" %d", gen_ids[i]);
    printf("\n");

    double prefill_s = prefill_t1 - prefill_t0;
    double decode_s = decode_t1 - decode_t0;
    double prefill_tok = (n_ids - 1 > 0) ? (double)(n_ids - 1) / prefill_s : 0.0;
    double decode_tok = (n_gen > 0 && decode_s > 0) ? (double)n_gen / decode_s : 0.0;

    long long hits, misses, bytes, stream_bytes;
    double io_ms, attn_ms, moe_ms, lm_head_ms;
    crom_stats(&m, &hits, &misses, &bytes, &stream_bytes, &io_ms,
               &attn_ms, &moe_ms, &lm_head_ms);
    long long total = hits + misses;
    double hit_pct = total > 0 ? 100.0 * (double)hits / (double)total : 100.0;

    FILE *f = fopen("/proc/self/status", "r");
    long rss_kb = 0;
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (!strncmp(line, "VmRSS:", 6)) { sscanf(line + 6, "%ld", &rss_kb); break; }
        }
        fclose(f);
    }

    printf("load_ms=%.1f prefill_tok/s=%.2f decode_tok/s=%.2f RSS_MB=%.1f "
           "cache_hit%%=%.2f preload_MB=%.2f ssd_stream_MB=%.2f ssd_io_ms=%.1f "
           "attn_ms=%.1f moe_ms=%.1f lm_head_ms=%.1f threads=%d cache=%d\n",
           load_ms, prefill_tok, decode_tok, (double)rss_kb / 1024.0,
           hit_pct, (double)bytes / (1024.0 * 1024.0), (double)stream_bytes / (1024.0 * 1024.0), io_ms,
           attn_ms, moe_ms, lm_head_ms, threads, m.cache_experts);

    crom_model_close(&m);
    free(hidden);
    free(hidden2);
    free(logits);
    free(ids);
    free(cur);
    free(gen_ids);
    return 0;
}
