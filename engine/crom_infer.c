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

static size_t mat_q4_bytes(int rows, int cols, int group_size) {
    return (size_t)rows * (cols / 2) + (size_t)rows * (cols / group_size) * 4;
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
/* Q4_g64 matvec AVX2+FMA (exato do bench_kernels.c kv_q4)           */
/* Layout: idx u8 [rows, cols/2]; byte c do grupo g =
 *   (q[c]+8) | ((q[c+32]+8)<<4); low nibble = peso g*64+c,
 *   high nibble = peso g*64+c+32. scales f32 [rows, cols/64].
 * w = (q-8)*scale.                                                   */
/* ------------------------------------------------------------------ */

static inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

static inline void i8_to_f32_32(__m256i v, __m256 *f) {
    __m256i a = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v));
    __m256i b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1));
    __m256i p[4] = {_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a)),
                    _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a, 1)),
                    _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b)),
                    _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b, 1))};
    for (int i = 0; i < 4; i++) f[i] = _mm256_cvtepi32_ps(p[i]);
}

static void q4_matvec(const crom_q4_mat_t *W, const float *x, float *out, int threads) {
    (void)threads;
    const int rows = W->rows;
    const int cols = W->cols;
    const int ng = cols / CROM_GROUP_SIZE;
    const int half = cols / 2;
    const uint8_t *idx = W->idx;
    const float *sc = W->scales;
    const __m256i maskF = _mm256_set1_epi8(0x0F);

#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const uint8_t *ip = idx + (size_t)r * half;
        const float *sp = sc + (size_t)r * ng;
        __m256 acc = _mm256_setzero_ps();
        for (int g = 0; g < ng; g++) {
            /* carrega 32 bytes do grupo g (idx u8) */
            __m256i p = _mm256_loadu_si256((const __m256i *)(ip + g * 32));
            __m256i lo = _mm256_sub_epi8(_mm256_and_si256(p, maskF), _mm256_set1_epi8(8));
            __m256i hi = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(p, 4), maskF), _mm256_set1_epi8(8));
            __m256 accg = _mm256_setzero_ps();
            const float *xg = x + g * CROM_GROUP_SIZE;
            for (int half = 0; half < 2; half++) {
                __m256 f[4];
                i8_to_f32_32(half ? hi : lo, f);
                for (int i = 0; i < 4; i++)
                    accg = _mm256_fmadd_ps(f[i], _mm256_loadu_ps(xg + half * 32 + i * 8), accg);
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(sp[g]), accg, acc);
        }
        out[r] = hsum256(acc);
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
    int total_experts;
    int *slot_of_expert;
    int *lru_order;
    int *pos_in_lru;
};

static struct lru_t *lru_create(int cap, int total_experts) {
    if (cap <= 0 || total_experts <= 0) return NULL;
    struct lru_t *l = calloc(1, sizeof(struct lru_t));
    if (!l) return NULL;
    l->cap = cap;
    l->total_experts = total_experts;
    l->slot_of_expert = calloc((size_t)total_experts, sizeof(int));
    if (!l->slot_of_expert) { free(l); return NULL; }
    for (int i = 0; i < total_experts; i++) l->slot_of_expert[i] = -1;
    l->lru_order = calloc((size_t)cap, sizeof(int));
    l->pos_in_lru = calloc((size_t)cap, sizeof(int));
    if (!l->lru_order || !l->pos_in_lru) {
        free(l->slot_of_expert);
        free(l->lru_order);
        free(l->pos_in_lru);
        free(l);
        return NULL;
    }
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
    if (e->size_bytes == 0) return -1;
    ssize_t rd = pread(m->fd, dst, (size_t)e->size_bytes, (off_t)e->offset_bytes);
    if (rd < 0) return -1;
    if ((size_t)rd != e->size_bytes) return -1;
    if (m->streaming) m->ssd_stream_bytes_read += (long long)e->size_bytes;
    else m->ssd_bytes_read += (long long)e->size_bytes;
    return 0;
}

static int lru_get_expert(crom_model_t *m, int expert_id, uint8_t **out_ptr,
                          double *io_ms) {
    struct lru_t *l = m->lru;
    int slot = l->slot_of_expert[expert_id];
    if (slot >= 0) {
        m->cache_hits++;
        lru_touch(l, slot);
        *out_ptr = (uint8_t *)m->expert_pool + (size_t)slot * m->expert_size;
        return 0;
    }
    m->cache_misses++;
    int evict_slot = l->lru_order[l->cap - 1];
    int evict_id = -1;
    for (int i = 0; i < l->total_experts; i++) {
        if (l->slot_of_expert[i] == evict_slot) { evict_id = i; break; }
    }
    if (evict_id >= 0) l->slot_of_expert[evict_id] = -1;

    double t0 = now_ms();
    uint8_t *dst = (uint8_t *)m->expert_pool + (size_t)evict_slot * m->expert_size;
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
    return crom_model_open_ctx(path, m, cache_experts, threads, 2048);
}

int crom_model_open_ctx(const char *path, crom_model_t *m, int cache_experts,
                        int threads, int max_seq) {
    memset(m, 0, sizeof(*m));
    m->threads = threads;
    m->max_seq = (max_seq > 0) ? max_seq : 2048;
    m->path = strdup(path);
    if (!m->path) { perror("strdup"); return -1; }

    int probe_fd = open(m->path, O_RDONLY);
    if (probe_fd < 0) { perror("open probe"); return -1; }

    struct stat st;
    if (fstat(probe_fd, &st) != 0) { perror("fstat"); close(probe_fd); return -1; }
    size_t fsize = (size_t)st.st_size;

    unsigned char hdr[CROM_HEADER_SIZE];
    if (pread(probe_fd, hdr, CROM_HEADER_SIZE, 0) != CROM_HEADER_SIZE) {
        fprintf(stderr, "leitura header falhou\n"); close(probe_fd); return -1;
    }
    close(probe_fd);

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

    if (m->version != CROM_VERSION || m->vocab_size == 0 ||
        m->hidden_size == 0 || m->num_layers == 0 ||
        m->num_attn_heads == 0 || m->num_kv_heads == 0 ||
        m->head_dim == 0 || m->num_local_experts == 0 ||
        m->num_experts_per_tok == 0 || m->group_size == 0) {
        fprintf(stderr, "dimensoes invalidas no header cromllm\n"); return -1;
    }

    m->total_experts = (int)(m->num_layers * m->num_local_experts);
    size_t exp_in = mat_q4_bytes(2 * (int)m->intermediate_size, (int)m->hidden_size, (int)m->group_size);
    size_t exp_out = mat_q4_bytes((int)m->hidden_size, (int)m->intermediate_size, (int)m->group_size);
    m->expert_size = exp_in + exp_out;

    if (cache_experts <= 0 || cache_experts > m->total_experts) {
        cache_experts = m->total_experts;
    }
    m->cache_experts = cache_experts;
    m->streaming = (cache_experts < m->total_experts);

    m->fd = open(m->path, m->streaming ? (O_RDONLY | O_DIRECT) : O_RDONLY);
    if (m->fd < 0) { perror("open fd"); return -1; }

    unsigned char *file_buf = NULL;
    if (!m->streaming) {
        file_buf = malloc(fsize);
        if (!file_buf) { perror("malloc file"); return -1; }
        ssize_t rd = pread(m->fd, file_buf, fsize, 0);
        if ((size_t)rd != fsize) { fprintf(stderr, "leitura incompleta\n"); return -1; }
    }

    m->hot_base = malloc(m->hot_size);
    if (!m->hot_base) { perror("malloc hot"); return -1; }
    if (file_buf) {
        memcpy(m->hot_base, file_buf + m->hot_offset, m->hot_size);
    } else {
        void *rbuf = aligned_alloc(CROM_PAGE, m->hot_size);
        if (!rbuf) { perror("aligned_alloc hot"); return -1; }
        if (pread(m->fd, rbuf, m->hot_size, (off_t)m->hot_offset) != (ssize_t)m->hot_size) {
            fprintf(stderr, "leitura hot falhou\n"); free(rbuf); return -1;
        }
        memcpy(m->hot_base, rbuf, m->hot_size);
        free(rbuf);
    }
    m->hot_len = m->hot_size;

    unsigned char *base = (unsigned char *)m->hot_base;
    size_t off = 0;

    /* embed [vocab, hidden] F16 (mantém em F16) */
    size_t embed_raw = (size_t)m->vocab_size * m->hidden_size * 2;
    size_t embed_pad = (embed_raw + CROM_PAGE - 1) & ~(CROM_PAGE - 1);
    m->embed_f16 = (uint16_t *)(base + off);
    off = embed_pad;

    /* norms [(num_layers * 2 + 1), hidden] FP32 */
    size_t num_norms = (size_t)(m->num_layers * 2 + 1);
    size_t norms_raw = num_norms * m->hidden_size * sizeof(float);
    size_t norms_pad = (norms_raw + CROM_PAGE - 1) & ~(CROM_PAGE - 1);
    m->norms_f32 = malloc(norms_raw);
    if (!m->norms_f32) { perror("malloc norms"); return -1; }
    memcpy(m->norms_f32, base + off, norms_raw);
    off += norms_pad;

    /* routers [num_layers, num_local_experts, hidden] FP32 */
    size_t routers_raw = (size_t)m->num_layers * m->num_local_experts * m->hidden_size * sizeof(float);
    size_t routers_pad = (routers_raw + CROM_PAGE - 1) & ~(CROM_PAGE - 1);
    m->routers_f32 = malloc(routers_raw);
    if (!m->routers_f32) { perror("malloc routers"); return -1; }
    memcpy(m->routers_f32, base + off, routers_raw);
    off += routers_pad;

    /* inv_freq [head_dim / 2] FP32 */
    size_t invf_count = m->head_dim / 2;
    size_t invf_raw = invf_count * sizeof(float);
    size_t invf_pad = (invf_raw + CROM_PAGE - 1) & ~(CROM_PAGE - 1);
    m->inv_freq = malloc(invf_raw);
    if (!m->inv_freq) { perror("malloc inv_freq"); return -1; }
    memcpy(m->inv_freq, base + off, invf_raw);
    off += invf_pad;

    /* atencao Q4: num_layers camadas q,k,v,o */
    m->layers = calloc(m->num_layers, sizeof(crom_layer_t));
    if (!m->layers) { perror("malloc layers"); return -1; }
    size_t qsz = mat_q4_bytes((int)(m->num_attn_heads * m->head_dim), (int)m->hidden_size, (int)m->group_size);
    size_t ksz = mat_q4_bytes((int)(m->num_kv_heads * m->head_dim), (int)m->hidden_size, (int)m->group_size);
    size_t osz = mat_q4_bytes((int)m->hidden_size, (int)(m->num_attn_heads * m->head_dim), (int)m->group_size);
    for (int li = 0; li < (int)m->num_layers; li++) {
        crom_layer_t *L = &m->layers[li];
        crom_attn_t *A = &L->attn;
        size_t ao = off;
        A->q_proj = q4_mat_from(base + ao, m->num_attn_heads * m->head_dim, m->hidden_size);
        ao += qsz;
        A->k_proj = q4_mat_from(base + ao, m->num_kv_heads * m->head_dim, m->hidden_size);
        ao += ksz;
        A->v_proj = q4_mat_from(base + ao, m->num_kv_heads * m->head_dim, m->hidden_size);
        ao += ksz;
        A->o_proj = q4_mat_from(base + ao, m->hidden_size, m->num_attn_heads * m->head_dim);
        ao += osz;
        off = ao;
        L->router_weight = m->routers_f32 + (size_t)li * m->num_local_experts * m->hidden_size;
        L->experts = calloc(m->num_local_experts, sizeof(crom_expert_t));
        if (!L->experts) { perror("calloc layer experts"); return -1; }
    }

    if (file_buf) free(file_buf);
    file_buf = NULL;

    /* Diretorio de experts */
    m->dir = calloc(m->total_experts, sizeof(crom_dir_entry_t));
    if (!m->dir) { perror("malloc dir"); return -1; }
    {
        size_t dir_sz = (size_t)m->total_experts * 24;
        size_t dir_pad = (dir_sz + CROM_PAGE - 1) & ~(CROM_PAGE - 1);
        void *dbuf = aligned_alloc(CROM_PAGE, dir_pad);
        if (!dbuf) { perror("aligned_alloc dir"); return -1; }
        if (pread(m->fd, dbuf, dir_sz, (off_t)m->expert_dir_offset) != (ssize_t)dir_sz) {
            fprintf(stderr, "leitura diretorio falhou (want=%zu off=%llu errno=%d)\n",
                    dir_sz, (unsigned long long)m->expert_dir_offset, errno);
            free(dbuf); return -1;
        }
        for (int i = 0; i < m->total_experts; i++) {
            const unsigned char *d = (const unsigned char *)dbuf + (size_t)i * 24;
            m->dir[i].offset_bytes = rd_u64(d + 0);
            m->dir[i].size_bytes = rd_u64(d + 8);
            m->dir[i].format_id = rd_u32(d + 16);
            m->dir[i].reserved = rd_u32(d + 20);
        }
        free(dbuf);
    }

    /* Pool de experts + LRU */
    m->expert_pool = aligned_alloc(CROM_PAGE, (size_t)cache_experts * m->expert_size);
    if (!m->expert_pool) { perror("aligned_alloc pool"); return -1; }
    m->lru = lru_create(cache_experts, m->total_experts);

    long long preload_bytes = 0;
    if (!m->streaming) {
        for (int i = 0; i < m->total_experts; i++) {
            uint8_t *dst = (uint8_t *)m->expert_pool + (size_t)i * m->expert_size;
            if (read_expert_payload(m, i, dst) != 0) {
                fprintf(stderr, "preload expert %d falhou\n", i); return -1;
            }
            preload_bytes += (long long)m->expert_size;
            m->lru->slot_of_expert[i] = i;
            m->lru->lru_order[i] = i;
            m->lru->pos_in_lru[i] = i;
            m->cache_hits++;
        }
    }
    m->ssd_bytes_read = preload_bytes;
    m->ssd_stream_bytes_read = 0;

    /* KV-cache */
    size_t kv_elems = (size_t)m->num_layers * m->num_kv_heads * m->max_seq * m->head_dim;
    m->k_cache = aligned_alloc(CROM_PAGE, kv_elems * sizeof(float));
    m->v_cache = aligned_alloc(CROM_PAGE, kv_elems * sizeof(float));
    if (!m->k_cache || !m->v_cache) { perror("aligned_alloc kv"); return -1; }
    memset(m->k_cache, 0, kv_elems * sizeof(float));
    memset(m->v_cache, 0, kv_elems * sizeof(float));

    /* Buffers de trabalho proporcionais */
    size_t max_dim = m->hidden_size;
    if (m->num_attn_heads * m->head_dim > max_dim) max_dim = m->num_attn_heads * m->head_dim;
    if (2 * (size_t)m->intermediate_size > max_dim) max_dim = 2 * (size_t)m->intermediate_size;

    m->work1 = aligned_alloc(CROM_PAGE, max_dim * sizeof(float) * 8);
    if (!m->work1) { perror("aligned_alloc work"); return -1; }

    omp_set_num_threads(threads);

    m->moe_acc_buf = aligned_alloc(CROM_PAGE, m->hidden_size * sizeof(float));
    if (!m->moe_acc_buf) { perror("aligned_alloc moe_acc"); return -1; }

    m->attn_scores = aligned_alloc(CROM_PAGE, (size_t)m->max_seq * sizeof(float));
    if (!m->attn_scores) { perror("aligned_alloc attn_scores"); return -1; }

    m->router_logits = malloc((size_t)m->num_local_experts * sizeof(float));
    m->router_idx = malloc((size_t)m->num_local_experts * sizeof(int));
    m->top_experts = malloc((size_t)m->num_experts_per_tok * sizeof(int));
    m->top_weights = malloc((size_t)m->num_experts_per_tok * sizeof(float));
    m->top_logits = malloc((size_t)m->num_experts_per_tok * sizeof(float));
    if (!m->router_logits || !m->router_idx || !m->top_experts || !m->top_weights || !m->top_logits) {
        perror("malloc routing buffers"); return -1;
    }

    m->attn_ms = 0.0;
    m->moe_ms = 0.0;
    m->lm_head_ms = 0.0;

    return 0;
}

#define KV_OFF(m, li, kvh, pos) \
    (((size_t)(li) * (size_t)(m)->num_kv_heads + (size_t)(kvh)) * (size_t)(m)->max_seq * (size_t)(m)->head_dim \
     + (size_t)(pos) * (size_t)(m)->head_dim)

/* ------------------------------------------------------------------ */
/* camada: input_layernorm + atencao + residual                       */
/* ------------------------------------------------------------------ */

int crom_layer_attn_resid(crom_model_t *m, int li, const float *hidden_in,
                          int pos, float *hidden_out) {
    double t0 = now_ms();
    int H = m->hidden_size;
    int head_dim = m->head_dim;
    int num_q_heads = m->num_attn_heads;
    int num_kv_heads = m->num_kv_heads;
    int kv_group = num_q_heads / num_kv_heads;

    size_t max_dim = H;
    if (m->num_attn_heads * m->head_dim > max_dim) max_dim = m->num_attn_heads * m->head_dim;
    if (2 * (size_t)m->intermediate_size > max_dim) max_dim = 2 * (size_t)m->intermediate_size;

    float *work1 = m->work1;
    float *work2 = work1 + max_dim;
    float *work3 = work2 + max_dim;
    float *work4 = work3 + max_dim;
    float *work5 = work4 + max_dim;

    float *rms_w_in = m->norms_f32 + (size_t)li * 2 * H;
    crom_rms_norm(rms_w_in, hidden_in, H, m->rms_norm_eps, work1);

    crom_attn_t *A = &m->layers[li].attn;
    q4_matvec(&A->q_proj, work1, work2, m->threads);
    q4_matvec(&A->k_proj, work1, work3, m->threads);
    q4_matvec(&A->v_proj, work1, work4, m->threads);

    for (int h = 0; h < num_q_heads; h++)
        rope_split_half(work2 + h * head_dim, pos, m->inv_freq, head_dim);
    for (int h = 0; h < num_kv_heads; h++)
        rope_split_half(work3 + h * head_dim, pos, m->inv_freq, head_dim);

    float *k_cache = m->k_cache;
    float *v_cache = m->v_cache;
    for (int h = 0; h < num_kv_heads; h++) {
        memcpy(k_cache + KV_OFF(m, li, h, pos), work3 + h * head_dim,
               head_dim * sizeof(float));
        memcpy(v_cache + KV_OFF(m, li, h, pos), work4 + h * head_dim,
               head_dim * sizeof(float));
    }

    int seqlen = pos + 1;
    if (seqlen > m->max_seq) seqlen = m->max_seq;
    float *attn_out = work5;
    float *scores = m->attn_scores;

    for (int h = 0; h < num_q_heads; h++) {
        int kvh = h / kv_group;
        float *q = work2 + h * head_dim;
        float maxs = -1e30f;
        for (int t = 0; t < seqlen; t++) {
            float s = 0.0f;
            const float *k = k_cache + KV_OFF(m, li, kvh, t);
            for (int d = 0; d < head_dim; d++) s += q[d] * k[d];
            s *= m->attention_multiplier;
            scores[t] = s;
            if (s > maxs) maxs = s;
        }
        float sum = 0.0f;
        for (int t = 0; t < seqlen; t++) { scores[t] = expf(scores[t] - maxs); sum += scores[t]; }
        float inv = 1.0f / sum;
        float *acc = attn_out + h * head_dim;
        for (int d = 0; d < head_dim; d++) acc[d] = 0.0f;
        for (int t = 0; t < seqlen; t++) {
            float w = scores[t] * inv;
            const float *v = v_cache + KV_OFF(m, li, kvh, t);
            for (int d = 0; d < head_dim; d++) acc[d] += v[d] * w;
        }
    }

    q4_matvec(&A->o_proj, attn_out, work1, m->threads);

    for (int i = 0; i < H; i++)
        hidden_out[i] = hidden_in[i] + work1[i] * m->residual_multiplier;
    
    m->attn_ms += now_ms() - t0;
    return 0;
}

/* ------------------------------------------------------------------ */
/* camada: post_attn_layernorm + moe + residual                       */
/* ------------------------------------------------------------------ */

int crom_layer_moe_resid(crom_model_t *m, int li, const float *hidden_in,
                          int pos, float *hidden_out,
                          int *top8_out, float *top8_w_out) {
    double t0 = now_ms();
    (void)pos;
    int H = m->hidden_size;
    int I = m->intermediate_size;
    int num_exp = m->num_local_experts;
    int num_top = m->num_experts_per_tok;

    size_t max_dim = H;
    if (m->num_attn_heads * m->head_dim > max_dim) max_dim = m->num_attn_heads * m->head_dim;
    if (2 * (size_t)I > max_dim) max_dim = 2 * (size_t)I;

    float *work1 = m->work1;
    float *work2 = work1 + max_dim;
    float *work3 = work2 + max_dim;
    float *work4 = work3 + max_dim;

    float *rms_w_post = m->norms_f32 + (size_t)li * 2 * H + H;
    crom_rms_norm(rms_w_post, hidden_in, H, m->rms_norm_eps, work1);

    float *router_w = m->layers[li].router_weight;
    float *router_logits = m->router_logits;
    int *router_idx = m->router_idx;
    for (int e = 0; e < num_exp; e++) {
        float s = 0.0f;
        const float *rw = router_w + (size_t)e * H;
        for (int i = 0; i < H; i++) s += work1[i] * rw[i];
        router_logits[e] = s;
        router_idx[e] = e;
    }

    for (int i = 0; i < num_top; i++) {
        int max_j = i;
        for (int j = i + 1; j < num_exp; j++) {
            if (router_logits[router_idx[j]] > router_logits[router_idx[max_j]]) {
                max_j = j;
            }
        }
        if (max_j != i) {
            int tmp = router_idx[i];
            router_idx[i] = router_idx[max_j];
            router_idx[max_j] = tmp;
        }
    }

    int *top_exp = m->top_experts;
    for (int i = 0; i < num_top; i++) top_exp[i] = router_idx[i];
    if (top8_out) memcpy(top8_out, top_exp, num_top * sizeof(int));

    float ml = -1e30f;
    for (int i = 0; i < num_top; i++) {
        float l = router_logits[top_exp[i]];
        if (l > ml) ml = l;
    }
    float *top_logits = m->top_logits;
    float *top_weights = m->top_weights;
    float sK = 0.0f;
    for (int i = 0; i < num_top; i++) {
        top_logits[i] = expf(router_logits[top_exp[i]] - ml);
        sK += top_logits[i];
    }
    float invK = 1.0f / sK;
    for (int i = 0; i < num_top; i++) {
        top_weights[i] = top_logits[i] * invK;
    }
    if (top8_w_out) memcpy(top8_w_out, top_weights, num_top * sizeof(float));

    float *moe_acc = m->moe_acc_buf;
    for (int i = 0; i < H; i++) moe_acc[i] = 0.0f;

    for (int ei = 0; ei < num_top; ei++) {
        int expert_id = top_exp[ei];
        float w = top_weights[ei];
        int dir_id = li * num_exp + expert_id;
        crom_expert_t *E = &m->layers[li].experts[expert_id];
        uint8_t *payload = NULL;
        if (m->streaming) {
            lru_get_expert(m, dir_id, &payload, NULL);
        } else {
            payload = (uint8_t *)m->expert_pool + (size_t)dir_id * m->expert_size;
        }
        size_t in_sz = mat_q4_bytes(2 * I, H, m->group_size);
        E->input_linear = q4_mat_from(payload, 2 * I, H);
        E->output_linear = q4_mat_from(payload + in_sz, H, I);

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
    
    m->moe_ms += now_ms() - t0;
    return 0;
}

/* ------------------------------------------------------------------ */
/* embed lookup (F16 -> FP32 on-the-fly com AVX2 _mm256_cvtph_ps)     */
/* ------------------------------------------------------------------ */

int crom_embed(crom_model_t *m, int token_id, float *hidden) {
    int H = m->hidden_size;
    const uint16_t *embed = m->embed_f16;
    const uint16_t *src = embed + (size_t)token_id * H;
    int i = 0;
    for (; i + 7 < H; i += 8) {
        __m128i f16 = _mm_loadu_si128((const __m128i *)(src + i));
        __m256 f32 = _mm256_cvtph_ps(f16);
        __m256 scaled = _mm256_mul_ps(f32, _mm256_set1_ps(m->embedding_multiplier));
        _mm256_storeu_ps(hidden + i, scaled);
    }
    for (; i < H; i++) {
        hidden[i] = _cvtsh_ss(src[i]) * m->embedding_multiplier;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* forward completo                                                   */
/* ------------------------------------------------------------------ */

int crom_forward(crom_model_t *m, const int *input_ids, int n_ids,
                 int position, float *hidden, float *out_logits) {
    (void)n_ids;
    int H = m->hidden_size;
    size_t max_dim = H;
    if (m->num_attn_heads * m->head_dim > max_dim) max_dim = m->num_attn_heads * m->head_dim;
    if (2 * (size_t)m->intermediate_size > max_dim) max_dim = 2 * (size_t)m->intermediate_size;

    float *work1 = m->work1;
    float *work2 = work1 + max_dim;

    crom_embed(m, input_ids[position], hidden);
    for (int li = 0; li < (int)m->num_layers; li++) {
        crom_layer_attn_resid(m, li, hidden, position, work2);
        crom_layer_moe_resid(m, li, work2, position, hidden, NULL, NULL);
    }

    /* final norm */
    float *final_norm_w = m->norms_f32 + (size_t)(m->num_layers * 2) * H;
    crom_rms_norm(final_norm_w, hidden, H, m->rms_norm_eps, work1);

    /* lm_head tied (F16C + FMA vector accumulator — hsum256 uma vez por linha) */
    double t0 = now_ms();
    int vocab = m->vocab_size;
    const uint16_t *embed = m->embed_f16;
#pragma omp parallel for schedule(static)
    for (int v = 0; v < vocab; v++) {
        __m256 vacc = _mm256_setzero_ps();
        const uint16_t *ev = embed + (size_t)v * H;
        const float *h = work1;
        int i = 0;
        for (; i + 7 < H; i += 8) {
            __m128i f16 = _mm_loadu_si128((const __m128i *)(ev + i));
            __m256 e = _mm256_cvtph_ps(f16);
            vacc = _mm256_fmadd_ps(e, _mm256_loadu_ps(h + i), vacc);
        }
        float s = hsum256(vacc);
        for (; i < H; i++) s += _cvtsh_ss(ev[i]) * h[i];
        out_logits[v] = s / m->logits_scaling;
    }
    m->lm_head_ms += now_ms() - t0;
    return 0;
}

/* tokenizer embutido - returns pointer to static buffer, caller must not free */
const void *crom_tokenizer_data(const crom_model_t *m, size_t *out_size) {
    if (out_size) *out_size = m->tokenizer_size;
    if (m->tokenizer_size == 0) return NULL;
    
    /* tokenizer.json está no offset tokenizer_offset do arquivo */
    static void *cached_buf = NULL;
    static size_t cached_size = 0;
    
    if (cached_buf && cached_size == m->tokenizer_size) {
        if (out_size) *out_size = cached_size;
        return cached_buf;
    }
    
    if (cached_buf) free(cached_buf);
    
    int fd = open(m->path, O_RDONLY);
    if (fd < 0) return NULL;
    
    void *buf = malloc(m->tokenizer_size);
    if (!buf) {
        close(fd);
        return NULL;
    }
    
    ssize_t rd = pread(fd, buf, m->tokenizer_size, (off_t)m->tokenizer_offset);
    close(fd);
    
    if (rd != (ssize_t)m->tokenizer_size) {
        free(buf);
        return NULL;
    }
    
    cached_buf = buf;
    cached_size = m->tokenizer_size;
    if (out_size) *out_size = cached_size;
    return cached_buf;
}

int crom_eos_id(void) {
    return 0;  /* <|end_of_text|> */
}

void crom_reset_kv(crom_model_t *m) {
    size_t kv_elems = (size_t)m->num_layers * m->num_kv_heads * m->max_seq * m->head_dim;
    memset(m->k_cache, 0, kv_elems * sizeof(float));
    memset(m->v_cache, 0, kv_elems * sizeof(float));
    m->attn_ms = 0.0;
    m->moe_ms = 0.0;
    m->lm_head_ms = 0.0;
}

/* ------------------------------------------------------------------ */
/* estatisticas                                                       */
/* ------------------------------------------------------------------ */

void crom_stats(const crom_model_t *m, long long *hits, long long *misses,
                long long *bytes, long long *stream_bytes, double *io_ms,
                double *attn_ms, double *moe_ms, double *lm_head_ms) {
    if (hits) *hits = m->cache_hits;
    if (misses) *misses = m->cache_misses;
    if (bytes) *bytes = m->ssd_bytes_read;
    if (stream_bytes) *stream_bytes = m->ssd_stream_bytes_read;
    if (io_ms) *io_ms = m->ssd_io_ms;
    if (attn_ms) *attn_ms = m->attn_ms;
    if (moe_ms) *moe_ms = m->moe_ms;
    if (lm_head_ms) *lm_head_ms = m->lm_head_ms;
}

void crom_model_close(crom_model_t *m) {
    if (m->lru) lru_destroy(m->lru);
    free(m->expert_pool);
    free(m->dir);
    free(m->hot_base);
    /* m->embed_f16 aponta para dentro de hot_base — NÃO free() */
    free(m->norms_f32);
    free(m->routers_f32);
    free(m->inv_freq);
    if (m->layers) {
        for (int li = 0; li < (int)m->num_layers; li++) {
            free(m->layers[li].experts);
        }
        free(m->layers);
    }
    free(m->k_cache);
    free(m->v_cache);
    free(m->work1);
    free(m->moe_acc_buf);
    free(m->attn_scores);
    free(m->router_logits);
    free(m->router_idx);
    free(m->top_experts);
    free(m->top_weights);
    free(m->top_logits);
    if (m->fd >= 0) close(m->fd);
    if (m->path) free((void*)m->path);
}

/* ------------------------------------------------------------------ */
/* Getters para Go/C                                                  */
/* ------------------------------------------------------------------ */

int crom_vocab_size(const crom_model_t *m) {
    return (int)m->vocab_size;
}

int crom_hidden_size(const crom_model_t *m) {
    return (int)m->hidden_size;
}

int crom_num_experts_total(const crom_model_t *m) {
    return m->total_experts;
}

int crom_max_seq(const crom_model_t *m) {
    return m->max_seq;
}
