#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <omp.h>
#include <immintrin.h>

#include "crom_infer.h"

/* ------------------------------------------------------------------ */
/* tempo                                                              */
/* ------------------------------------------------------------------ */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* ------------------------------------------------------------------ */
/* leitura little-endian                                              */
/* ------------------------------------------------------------------ */

static uint32_t rd_u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t rd_u64(const unsigned char *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

static size_t mat_q4_bytes(int rows, int cols) {
    return (size_t)rows * (cols / 2) + (size_t)rows * (cols / CROM_GROUP_SIZE) * 4;
}

static crom_q4_mat_t q4_mat_from(const unsigned char *p, int rows, int cols) {
    crom_q4_mat_t W;
    W.rows = rows; W.cols = cols;
    size_t idx_sz = (size_t)rows * (cols / 2);
    W.idx = (uint8_t *)p;
    W.scales = (float *)((const unsigned char *)p + idx_sz);
    return W;
}

/* ------------------------------------------------------------------ */
/* Q4_g64 matvec (escalar por linha, OpenMP nas linhas)               */
/* Layout: idx u8 [rows, cols/2]; byte c do grupo g =
 *   (q[c]+8) | ((q[c+32]+8)<<4); low nibble = peso g*64+c,
 *   high nibble = peso g*64+c+32. scales f32 [rows, cols/64].
 * w = (q-8)*scale.                                                   */
/* ------------------------------------------------------------------ */

static void q4_matvec(const crom_q4_mat_t *W, const float *x, float *out, int threads) {
    (void)threads;
    const int rows = W->rows;
    const int cols = W->cols;
    const int ng = cols / CROM_GROUP_SIZE;
    const int half = cols / 2;
    const uint8_t *idx = W->idx;
    const float *sc = W->scales;

#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        const uint8_t *idx_r = idx + (size_t)r * half;
        const float *sc_r = sc + (size_t)r * ng;
        for (int g = 0; g < ng; g++) {
            float s = sc_r[g];
            for (int c = 0; c < 32; c++) {
                uint8_t byte = idx_r[g * 32 + c];
                int qlo = (int)(byte & 0x0F) - 8;
                int qhi = (int)((byte >> 4) & 0x0F) - 8;
                acc += x[g * 64 + c] * (float)qlo * s;
                acc += x[g * 64 + c + 32] * (float)qhi * s;
            }
        }
        out[r] = acc;
    }
}

/* ------------------------------------------------------------------ */
/* RMSNorm FP32 (exportado)                                           */
/* ------------------------------------------------------------------ */

void crom_rms_norm(const float *w, const float *x, int n, float eps, float *out) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

/* ------------------------------------------------------------------ */
/* RoPE split-half                                                    */
/* ------------------------------------------------------------------ */

static void rope_split_half(float *x, int pos, const float *inv_freq, int head_dim) {
    const int half = head_dim / 2;
    for (int j = 0; j < half; j++) {
        float angle = (float)pos * inv_freq[j];
        float c = cosf(angle), s = sinf(angle);
        float x1 = x[j];
        float x2 = x[j + half];
        x[j] = x1 * c - x2 * s;
        x[j + half] = x1 * s + x2 * c;
    }
}

/* ------------------------------------------------------------------ */
/* SiLU                                                               */
/* ------------------------------------------------------------------ */

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

/* ------------------------------------------------------------------ */
/* LRU cache O(1)                                                     */
/* ------------------------------------------------------------------ */

struct lru_t {
    int cap;
    int *slot_of_expert;
    int *lru_order;
    int *pos_in_lru;
};

static struct lru_t *lru_create(int cap) {
    struct lru_t *l = calloc(1, sizeof(struct lru_t));
    l->cap = cap;
    l->slot_of_expert = calloc(CROM_NUM_DIR_ENTRIES, sizeof(int));
    for (int i = 0; i < CROM_NUM_DIR_ENTRIES; i++) l->slot_of_expert[i] = -1;
    l->lru_order = calloc(cap, sizeof(int));
    l->pos_in_lru = calloc(cap, sizeof(int));
    for (int i = 0; i < cap; i++) { l->lru_order[i] = i; l->pos_in_lru[i] = i; }
    return l;
}

static void lru_touch(struct lru_t *l, int slot) {
    int pos = l->pos_in_lru[slot];
    int id = l->lru_order[pos];
    for (int p = pos; p > 0; p--) {
        l->lru_order[p] = l->lru_order[p - 1];
        l->pos_in_lru[l->lru_order[p]] = p;
    }
    l->lru_order[0] = id;
    l->pos_in_lru[id] = 0;
}

static void lru_destroy(struct lru_t *l) {
    free(l->slot_of_expert);
    free(l->lru_order);
    free(l->pos_in_lru);
    free(l);
}

/* ------------------------------------------------------------------ */
/* leitura de expert via pread alinhada                               */
/* ------------------------------------------------------------------ */

static int read_expert_payload(crom_model_t *m, int expert_id, uint8_t *dst) {
    const crom_dir_entry_t *e = &m->dir[expert_id];
    if (e->size_bytes != CROM_EXPERT_SIZE) return -1;
    ssize_t rd = pread(m->fd, dst, (size_t)e->size_bytes, (off_t)e->offset_bytes);
    if (rd < 0) return -1;
    if ((size_t)rd != e->size_bytes) return -1;
    m->ssd_bytes_read += (long long)e->size_bytes;
    return 0;
}

static int lru_get_expert(crom_model_t *m, int expert_id, uint8_t **out_ptr,
                          double *io_ms) {
    struct lru_t *l = m->lru;
    int slot = l->slot_of_expert[expert_id];
    if (slot >= 0) {
        m->cache_hits++;
        lru_touch(l, slot);
        *out_ptr = (uint8_t *)m->expert_pool + (size_t)slot * CROM_EXPERT_SIZE;
        return 0;
    }
    m->cache_misses++;
    int evict_slot = l->lru_order[l->cap - 1];
    int evict_id = -1;
    for (int i = 0; i < CROM_NUM_DIR_ENTRIES; i++) {
        if (l->slot_of_expert[i] == evict_slot) { evict_id = i; break; }
    }
    if (evict_id >= 0) l->slot_of_expert[evict_id] = -1;

    double t0 = now_ms();
    uint8_t *dst = (uint8_t *)m->expert_pool + (size_t)evict_slot * CROM_EXPERT_SIZE;
    if (read_expert_payload(m, expert_id, dst) != 0) return -1;
    double t1 = now_ms();
    m->ssd_io_ms += (t1 - t0);
    if (io_ms) *io_ms += (t1 - t0);

    l->slot_of_expert[expert_id] = evict_slot;
    lru_touch(l, evict_slot);
    *out_ptr = dst;
    return 0;
}

/* ------------------------------------------------------------------ */
/* abrir modelo                                                       */
/* ------------------------------------------------------------------ */

int crom_model_open(const char *path, crom_model_t *m, int cache_experts,
                    int threads) {
    memset(m, 0, sizeof(*m));
    m->threads = threads;
    m->cache_experts = cache_experts;
    m->streaming = (cache_experts < CROM_NUM_DIR_ENTRIES);
    m->fd = open(path, m->streaming ? (O_RDONLY | O_DIRECT) : O_RDONLY);
    if (m->fd < 0) { perror("open"); return -1; }

    struct stat st;
    if (fstat(m->fd, &st) != 0) { perror("fstat"); return -1; }
    size_t fsize = (size_t)st.st_size;

    unsigned char *file_buf = NULL;
    if (!m->streaming) {
        file_buf = malloc(fsize);
        if (!file_buf) { perror("malloc file"); return -1; }
        ssize_t rd = read(m->fd, file_buf, fsize);
        if ((size_t)rd != fsize) { fprintf(stderr, "leitura incompleta\n"); return -1; }
    }

    unsigned char hdr[CROM_HEADER_SIZE];
    if (file_buf) {
        memcpy(hdr, file_buf, CROM_HEADER_SIZE);
    } else {
        unsigned char *page = aligned_alloc(CROM_PAGE, CROM_PAGE);
        if (!page) { perror("aligned_alloc header"); return -1; }
        if (pread(m->fd, page, CROM_PAGE, 0) != CROM_PAGE) {
            free(page);
            fprintf(stderr, "leitura header falhou\n"); return -1;
        }
        memcpy(hdr, page, CROM_HEADER_SIZE);
        free(page);
    }
    memcpy(m->magic_check, hdr, 8);
    if (memcmp(m->magic_check, CROM_MAGIC, 8) != 0) {
        fprintf(stderr, "magic invalido: %.8s\n", m->magic_check); return -1;
    }
    const unsigned char *hh = hdr;
    m->version = rd_u32(hh + 8);
    m->vocab_size = rd_u32(hh + 12);
    m->hidden_size = rd_u32(hh + 16);
    m->intermediate_size = rd_u32(hh + 20);
    m->num_layers = rd_u32(hh + 24);
    m->num_attn_heads = rd_u32(hh + 28);
    m->num_kv_heads = rd_u32(hh + 32);
    m->num_local_experts = rd_u32(hh + 36);
    m->num_experts_per_tok = rd_u32(hh + 40);
    m->head_dim = rd_u32(hh + 44);
    m->embed_format = rd_u32(hh + 48);
    m->attn_format = rd_u32(hh + 52);
    m->expert_format = rd_u32(hh + 56);
    m->group_size = rd_u32(hh + 60);
    memcpy(&m->rope_theta, hh + 64, 4);
    memcpy(&m->rms_norm_eps, hh + 68, 4);
    memcpy(&m->embedding_multiplier, hh + 72, 4);
    memcpy(&m->attention_multiplier, hh + 76, 4);
    memcpy(&m->residual_multiplier, hh + 80, 4);
    memcpy(&m->logits_scaling, hh + 84, 4);
    m->hot_offset = rd_u64(hh + 88);
    m->hot_size = rd_u64(hh + 96);
    m->expert_dir_offset = rd_u32(hh + 104);
    m->expert_dir_count = rd_u32(hh + 108);
    m->expert_dir_entry_size = rd_u64(hh + 112);
    m->experts_offset = rd_u64(hh + 120);
    m->tokenizer_offset = rd_u64(hh + 128);
    m->tokenizer_size = rd_u64(hh + 136);

    if (m->version != CROM_VERSION || m->vocab_size != CROM_VOCAB_SIZE ||
        m->hidden_size != CROM_HIDDEN_SIZE || m->num_layers != CROM_NUM_LAYERS ||
        m->num_attn_heads != CROM_NUM_Q_HEADS || m->num_kv_heads != CROM_NUM_KV_HEADS ||
        m->head_dim != CROM_HEAD_DIM || m->num_local_experts != CROM_NUM_EXPERTS ||
        m->group_size != CROM_GROUP_SIZE) {
        fprintf(stderr, "dimensoes nao batem com spec\n"); return -1;
    }

    m->hot_base = malloc(m->hot_size);
    if (!m->hot_base) { perror("malloc hot"); return -1; }
    if (file_buf) {
        memcpy(m->hot_base, file_buf + m->hot_offset, m->hot_size);
    } else {
        void *rbuf = aligned_alloc(CROM_PAGE, m->hot_size);
        if (!rbuf) { perror("aligned_alloc hot"); return -1; }
        if (pread(m->fd, rbuf, m->hot_size, (off_t)m->hot_offset) != (ssize_t)m->hot_size) {
            fprintf(stderr, "leitura hot falhou\n"); return -1;
        }
        memcpy(m->hot_base, rbuf, m->hot_size);
        free(rbuf);
    }
    m->hot_len = m->hot_size;

    unsigned char *base = (unsigned char *)m->hot_base;
    size_t off = 0;

    /* embed [vocab, hidden] F16 -> FP32 */
    size_t embed_raw = (size_t)CROM_VOCAB_SIZE * CROM_HIDDEN_SIZE * 2;
    size_t embed_pad = (embed_raw + CROM_PAGE - 1) & ~(CROM_PAGE - 1);
    m->embed_f32 = malloc((size_t)CROM_VOCAB_SIZE * CROM_HIDDEN_SIZE * sizeof(float));
    if (!m->embed_f32) { perror("malloc embed"); return -1; }
    {
        const uint16_t *f16 = (const uint16_t *)(base + off);
        float *f32 = m->embed_f32;
        size_t ne = (size_t)CROM_VOCAB_SIZE * CROM_HIDDEN_SIZE;
        for (size_t i = 0; i < ne; i++) f32[i] = _cvtsh_ss(f16[i]);
    }
    off = embed_pad;

    /* norms [49, hidden] FP32 */
    size_t norms_raw = (size_t)49 * CROM_HIDDEN_SIZE * sizeof(float);
    size_t norms_pad = (norms_raw + CROM_PAGE - 1) & ~(CROM_PAGE - 1);
    m->norms_f32 = malloc(norms_raw);
    if (!m->norms_f32) { perror("malloc norms"); return -1; }
    memcpy(m->norms_f32, base + off, norms_raw);
    off += norms_pad;

    /* routers [24, 32, 1024] FP32 */
    size_t routers_raw = (size_t)24 * 32 * CROM_HIDDEN_SIZE * sizeof(float);
    size_t routers_pad = (routers_raw + CROM_PAGE - 1) & ~(CROM_PAGE - 1);
    m->routers_f32 = malloc(routers_raw);
    if (!m->routers_f32) { perror("malloc routers"); return -1; }
    memcpy(m->routers_f32, base + off, routers_raw);
    off += routers_pad;

    /* inv_freq [32] FP32 */
    size_t invf_raw = 32 * sizeof(float);
    size_t invf_pad = (invf_raw + CROM_PAGE - 1) & ~(CROM_PAGE - 1);
    m->inv_freq = malloc(32 * sizeof(float));
    if (!m->inv_freq) { perror("malloc inv_freq"); return -1; }
    memcpy(m->inv_freq, base + off, invf_raw);
    off += invf_pad;

    /* atencao Q4: 24 camadas q,k,v,o */
    m->layers = calloc(m->num_layers, sizeof(crom_layer_t));
    if (!m->layers) { perror("malloc layers"); return -1; }
    size_t qsz = mat_q4_bytes(CROM_HIDDEN_SIZE, CROM_HIDDEN_SIZE);
    size_t ksz = mat_q4_bytes(CROM_NUM_KV_HEADS * CROM_HEAD_DIM, CROM_HIDDEN_SIZE);
    for (int li = 0; li < (int)m->num_layers; li++) {
        crom_layer_t *L = &m->layers[li];
        crom_attn_t *A = &L->attn;
        size_t ao = off;
        A->q_proj = q4_mat_from(base + ao, CROM_HIDDEN_SIZE, CROM_HIDDEN_SIZE);
        ao += qsz;
        A->k_proj = q4_mat_from(base + ao, CROM_NUM_KV_HEADS * CROM_HEAD_DIM, CROM_HIDDEN_SIZE);
        ao += ksz;
        A->v_proj = q4_mat_from(base + ao, CROM_NUM_KV_HEADS * CROM_HEAD_DIM, CROM_HIDDEN_SIZE);
        ao += ksz;
        A->o_proj = q4_mat_from(base + ao, CROM_HIDDEN_SIZE, CROM_HIDDEN_SIZE);
        ao += qsz;
        off = ao;
        L->router_weight = m->routers_f32 + (size_t)li * 32 * CROM_HIDDEN_SIZE;
    }

    if (file_buf) free(file_buf);
    file_buf = NULL;

    /* Diretorio de experts */
    m->dir = calloc(CROM_NUM_DIR_ENTRIES, sizeof(crom_dir_entry_t));
    if (!m->dir) { perror("malloc dir"); return -1; }
    {
        size_t dir_sz = (size_t)CROM_NUM_DIR_ENTRIES * CROM_DIR_ENTRY_SIZE;
        size_t dir_pad = (dir_sz + CROM_PAGE - 1) & ~(CROM_PAGE - 1);
        void *dbuf = aligned_alloc(CROM_PAGE, dir_pad);
        if (!dbuf) { perror("aligned_alloc dir"); return -1; }
        if (pread(m->fd, dbuf, dir_sz, (off_t)m->expert_dir_offset) != (ssize_t)dir_sz) {
            fprintf(stderr, "leitura diretorio falhou (want=%zu off=%llu errno=%d)\n",
                    dir_sz, (unsigned long long)m->expert_dir_offset, errno);
            free(dbuf); return -1;
        }
        for (int i = 0; i < CROM_NUM_DIR_ENTRIES; i++) {
            const unsigned char *d = (const unsigned char *)dbuf + (size_t)i * 24;
            m->dir[i].offset_bytes = rd_u64(d + 0);
            m->dir[i].size_bytes = rd_u64(d + 8);
            m->dir[i].format_id = rd_u32(d + 16);
            m->dir[i].reserved = rd_u32(d + 20);
        }
        free(dbuf);
    }

    /* Pool de experts + LRU */
    m->expert_pool = aligned_alloc(CROM_PAGE, (size_t)cache_experts * CROM_EXPERT_SIZE);
    if (!m->expert_pool) { perror("aligned_alloc pool"); return -1; }
    m->lru = lru_create(cache_experts);

    if (!m->streaming) {
        for (int i = 0; i < CROM_NUM_DIR_ENTRIES; i++) {
            uint8_t *dst = (uint8_t *)m->expert_pool + (size_t)i * CROM_EXPERT_SIZE;
            if (read_expert_payload(m, i, dst) != 0) {
                fprintf(stderr, "preload expert %d falhou\n", i); return -1;
            }
            m->lru->slot_of_expert[i] = i;
            m->lru->lru_order[i] = i;
            m->lru->pos_in_lru[i] = i;
            m->cache_hits++;
        }
    }

    /* KV-cache */
    size_t kv_elems = (size_t)CROM_NUM_LAYERS * CROM_NUM_KV_HEADS * CROM_MAX_SEQ * CROM_HEAD_DIM;
    m->k_cache = aligned_alloc(CROM_PAGE, kv_elems * sizeof(float));
    m->v_cache = aligned_alloc(CROM_PAGE, kv_elems * sizeof(float));
    if (!m->k_cache || !m->v_cache) { perror("aligned_alloc kv"); return -1; }
    memset(m->k_cache, 0, kv_elems * sizeof(float));
    memset(m->v_cache, 0, kv_elems * sizeof(float));

    /* Buffers de trabalho */
    m->work1 = aligned_alloc(CROM_PAGE, (size_t)CROM_HIDDEN_SIZE * sizeof(float) * 8);
    if (!m->work1) { perror("aligned_alloc work"); return -1; }

    omp_set_num_threads(threads);

    /* buffer dedicado p/ moe_acc (evita colisao com hidden_in) */
    size_t wb = (size_t)CROM_HIDDEN_SIZE;
    m->moe_acc_buf = aligned_alloc(CROM_PAGE, wb * sizeof(float));
    if (!m->moe_acc_buf) { perror("aligned_alloc moe_acc"); return -1; }

    return 0;
}

#define KV_OFF(m, li, kvh, pos) \
    (((size_t)(li) * CROM_NUM_KV_HEADS + (kvh)) * (size_t)CROM_MAX_SEQ * CROM_HEAD_DIM \
     + (size_t)(pos) * CROM_HEAD_DIM)

/* ------------------------------------------------------------------ */
/* camada: input_layernorm + atencao + residual                       */
/* ------------------------------------------------------------------ */

int crom_layer_attn_resid(crom_model_t *m, int li, const float *hidden_in,
                          int pos, float *hidden_out) {
    int H = CROM_HIDDEN_SIZE;
    float *work1 = m->work1;
    float *work2 = work1 + H;
    float *work3 = work2 + H;
    float *work4 = work3 + H;
    float *work5 = work4 + H;

    float *rms_w_in = m->norms_f32 + (size_t)li * 2 * H;
    crom_rms_norm(rms_w_in, hidden_in, H, m->rms_norm_eps, work1);

    crom_attn_t *A = &m->layers[li].attn;
    q4_matvec(&A->q_proj, work1, work2, m->threads);
    q4_matvec(&A->k_proj, work1, work3, m->threads);
    q4_matvec(&A->v_proj, work1, work4, m->threads);

    for (int h = 0; h < CROM_NUM_Q_HEADS; h++)
        rope_split_half(work2 + h * CROM_HEAD_DIM, pos, m->inv_freq, CROM_HEAD_DIM);
    for (int h = 0; h < CROM_NUM_KV_HEADS; h++)
        rope_split_half(work3 + h * CROM_HEAD_DIM, pos, m->inv_freq, CROM_HEAD_DIM);

    float *k_cache = m->k_cache;
    float *v_cache = m->v_cache;
    for (int h = 0; h < CROM_NUM_KV_HEADS; h++) {
        memcpy(k_cache + KV_OFF(m, li, h, pos), work3 + h * CROM_HEAD_DIM,
               CROM_HEAD_DIM * sizeof(float));
        memcpy(v_cache + KV_OFF(m, li, h, pos), work4 + h * CROM_HEAD_DIM,
               CROM_HEAD_DIM * sizeof(float));
    }

    int seqlen = pos + 1;
    float *attn_out = work5;
    for (int h = 0; h < CROM_NUM_Q_HEADS; h++) {
        int kvh = h / 2;
        float *q = work2 + h * CROM_HEAD_DIM;
        float scores[CROM_MAX_SEQ];
        float maxs = -1e30f;
        for (int t = 0; t < seqlen; t++) {
            float s = 0.0f;
            const float *k = k_cache + KV_OFF(m, li, kvh, t);
            for (int d = 0; d < CROM_HEAD_DIM; d++) s += q[d] * k[d];
            s *= m->attention_multiplier;
            scores[t] = s;
            if (s > maxs) maxs = s;
        }
        float sum = 0.0f;
        for (int t = 0; t < seqlen; t++) { scores[t] = expf(scores[t] - maxs); sum += scores[t]; }
        float inv = 1.0f / sum;
        float *acc = attn_out + h * CROM_HEAD_DIM;
        for (int d = 0; d < CROM_HEAD_DIM; d++) acc[d] = 0.0f;
        for (int t = 0; t < seqlen; t++) {
            float w = scores[t] * inv;
            const float *v = v_cache + KV_OFF(m, li, kvh, t);
            for (int d = 0; d < CROM_HEAD_DIM; d++) acc[d] += v[d] * w;
        }
    }

    q4_matvec(&A->o_proj, attn_out, work1, m->threads);

    for (int i = 0; i < H; i++)
        hidden_out[i] = hidden_in[i] + work1[i] * m->residual_multiplier;
    return 0;
}

/* ------------------------------------------------------------------ */
/* camada: post_attn_layernorm + moe + residual                       */
/* ------------------------------------------------------------------ */

int crom_layer_moe_resid(crom_model_t *m, int li, const float *hidden_in,
                         int pos, float *hidden_out,
                         int *top8_out, float *top8_w_out) {
    (void)pos;
    int H = CROM_HIDDEN_SIZE;
    int I = CROM_INTER_SIZE;
    float *work1 = m->work1;
    float *work2 = work1 + H;
    float *work3 = work2 + H;
    float *work4 = work3 + H;

    float *rms_w_post = m->norms_f32 + (size_t)li * 2 * H + H;
    crom_rms_norm(rms_w_post, hidden_in, H, m->rms_norm_eps, work1);

    float *router_w = m->layers[li].router_weight;
    float router_logits[CROM_NUM_EXPERTS];
    for (int e = 0; e < CROM_NUM_EXPERTS; e++) {
        float s = 0.0f;
        const float *rw = router_w + (size_t)e * H;
        for (int i = 0; i < H; i++) s += work1[i] * rw[i];
        router_logits[e] = s;
    }
    int idx[CROM_NUM_EXPERTS];
    for (int e = 0; e < CROM_NUM_EXPERTS; e++) idx[e] = e;
    for (int i = 0; i < CROM_NUM_EXPERTS; i++)
        for (int j = i + 1; j < CROM_NUM_EXPERTS; j++)
            if (router_logits[idx[j]] > router_logits[idx[i]]) {
                int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            }
    int top8[CROM_EXPERTS_PER_TOK];
    for (int i = 0; i < CROM_EXPERTS_PER_TOK; i++) top8[i] = idx[i];
    if (top8_out) memcpy(top8_out, top8, sizeof(top8));

    float ml = -1e30f;
    for (int i = 0; i < CROM_EXPERTS_PER_TOK; i++)
        if (router_logits[top8[i]] > ml) ml = router_logits[top8[i]];
    float top8_logits[CROM_EXPERTS_PER_TOK];
    float s8 = 0.0f;
    for (int i = 0; i < CROM_EXPERTS_PER_TOK; i++) {
        top8_logits[i] = expf(router_logits[top8[i]] - ml);
        s8 += top8_logits[i];
    }
    float inv8 = 1.0f / s8;
    float w8[CROM_EXPERTS_PER_TOK];
    for (int i = 0; i < CROM_EXPERTS_PER_TOK; i++)
        w8[i] = top8_logits[i] * inv8;
    if (top8_w_out) memcpy(top8_w_out, w8, sizeof(w8));

    float *moe_acc = m->moe_acc_buf;
    for (int i = 0; i < H; i++) moe_acc[i] = 0.0f;

    for (int ei = 0; ei < CROM_EXPERTS_PER_TOK; ei++) {
        int expert_id = top8[ei];
        float w = w8[ei];
        int dir_id = li * CROM_NUM_EXPERTS + expert_id;
        crom_expert_t *E = &m->layers[li].experts[expert_id];
        uint8_t *payload = NULL;
        if (m->streaming) {
            lru_get_expert(m, dir_id, &payload, NULL);
        } else {
            payload = (uint8_t *)m->expert_pool + (size_t)dir_id * CROM_EXPERT_SIZE;
        }
        size_t in_idx = (size_t)H * (H / 2);
        E->input_linear = q4_mat_from(payload, H, H);
        size_t out_idx_off = in_idx + (size_t)H * (H / CROM_GROUP_SIZE) * sizeof(float);
        E->output_linear = q4_mat_from(payload + out_idx_off, H, I);

        q4_matvec(&E->input_linear, work1, work3, m->threads);
        for (int i = 0; i < I; i++) {
            float g = silu(work3[i]);
            float u = work3[I + i];
            work3[i] = g * u;
        }
        q4_matvec(&E->output_linear, work3, work4, m->threads);
        for (int i = 0; i < H; i++) moe_acc[i] += work4[i] * w;
    }

    for (int i = 0; i < H; i++)
        hidden_out[i] = hidden_in[i] + moe_acc[i] * m->residual_multiplier;
    return 0;
}

/* ------------------------------------------------------------------ */
/* embed lookup                                                       */
/* ------------------------------------------------------------------ */

int crom_embed(crom_model_t *m, int token_id, float *hidden) {
    int H = CROM_HIDDEN_SIZE;
    const float *embed = m->embed_f32;
    for (int i = 0; i < H; i++)
        hidden[i] = embed[(size_t)token_id * H + i] * m->embedding_multiplier;
    return 0;
}

/* ------------------------------------------------------------------ */
/* forward completo                                                   */
/* ------------------------------------------------------------------ */

int crom_forward(crom_model_t *m, const int *input_ids, int n_ids,
                 int position, float *hidden, float *out_logits) {
    (void)n_ids;
    int H = CROM_HIDDEN_SIZE;
    float *work1 = m->work1;
    float *work2 = work1 + H;

crom_embed(m, input_ids[position], hidden);
    for (int li = 0; li < (int)m->num_layers; li++) {
        crom_layer_attn_resid(m, li, hidden, position, work2);
        crom_layer_moe_resid(m, li, work2, position, hidden, NULL, NULL);
    }

    /* final norm */
    float *final_norm_w = m->norms_f32 + (size_t)(m->num_layers * 2) * H;
    crom_rms_norm(final_norm_w, hidden, H, m->rms_norm_eps, work1);

    /* lm_head tied */
    int vocab = CROM_VOCAB_SIZE;
    const float *embed = m->embed_f32;
#pragma omp parallel for schedule(static)
    for (int v = 0; v < vocab; v++) {
        float s = 0.0f;
        const float *ev = embed + (size_t)v * H;
        const float *h = work1;
        int i = 0;
        for (; i + 7 < H; i += 8) {
            __m256 e = _mm256_loadu_ps(ev + i);
            __m256 hh = _mm256_loadu_ps(h + i);
            __m256 p = _mm256_mul_ps(e, hh);
            const float *pf = (const float *)&p;
            s += pf[0] + pf[1] + pf[2] + pf[3] + pf[4] + pf[5] + pf[6] + pf[7];
        }
        for (; i < H; i++) s += ev[i] * h[i];
        out_logits[v] = s / m->logits_scaling;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* estatisticas                                                       */
/* ------------------------------------------------------------------ */

void crom_stats(const crom_model_t *m, long long *hits, long long *misses,
                long long *bytes, double *io_ms) {
    if (hits) *hits = m->cache_hits;
    if (misses) *misses = m->cache_misses;
    if (bytes) *bytes = m->ssd_bytes_read;
    if (io_ms) *io_ms = m->ssd_io_ms;
}

void crom_model_close(crom_model_t *m) {
    if (m->lru) lru_destroy(m->lru);
    free(m->expert_pool);
    free(m->dir);
    free(m->hot_base);
    free(m->embed_f32);
    free(m->norms_f32);
    free(m->routers_f32);
    free(m->inv_freq);
    free(m->layers);
    free(m->k_cache);
    free(m->v_cache);
    free(m->work1);
    free(m->moe_acc_buf);
    if (m->fd >= 0) close(m->fd);
}
