#ifndef CROM_INFER_H
#define CROM_INFER_H

#include <stdint.h>
#include <stddef.h>

/* ============ Constantes do formato .cromllm (docs/03-formato-cromllm.md) ============ */

#define CROM_MAGIC   "CROMLM01"
#define CROM_VERSION 1
#define CROM_HEADER_SIZE 144
#define CROM_PAGE 4096

#define CROM_VOCAB_SIZE   49155
#define CROM_HIDDEN_SIZE  1024
#define CROM_INTER_SIZE   512
#define CROM_NUM_LAYERS   24
#define CROM_NUM_Q_HEADS  16
#define CROM_NUM_KV_HEADS 8
#define CROM_HEAD_DIM     64
#define CROM_NUM_EXPERTS  32
#define CROM_EXPERTS_PER_TOK 8
#define CROM_GROUP_SIZE   64

#define CROM_EXPERT_SIZE  884736
#define CROM_NUM_DIR_ENTRIES (CROM_NUM_LAYERS * CROM_NUM_EXPERTS) /* 768 */
#define CROM_DIR_ENTRY_SIZE 24
#define CROM_HOT_SIZE     146489344
#define CROM_MAX_SEQ      512

#define CROM_RMS_NORM_EPS     1e-6f
#define CROM_EMBED_MULTIPLIER 12.0f
#define CROM_ATTN_MULTIPLIER  0.015625f
#define CROM_RESID_MULTIPLIER 0.22f
#define CROM_LOGITS_SCALING   6.0f

struct lru_t;

typedef struct {
    uint64_t offset_bytes;
    uint64_t size_bytes;
    uint32_t format_id;
    uint32_t reserved;
} crom_dir_entry_t;

typedef struct {
    uint8_t  *idx;
    float    *scales;
    int rows;
    int cols;
} crom_q4_mat_t;

typedef struct {
    crom_q4_mat_t q_proj, k_proj, v_proj, o_proj;
} crom_attn_t;

typedef struct {
    crom_q4_mat_t input_linear;
    crom_q4_mat_t output_linear;
} crom_expert_t;

typedef struct {
    crom_attn_t  attn;
    crom_expert_t *experts;  /* alocado dinamicamente: num_local_experts */
    float *router_weight;   /* aponta para routers_f32 */
} crom_layer_t;

typedef struct {
    unsigned char magic_check[8];
    uint32_t version;
    uint32_t vocab_size, hidden_size, intermediate_size, num_layers;
    uint32_t num_attn_heads, num_kv_heads, num_local_experts, num_experts_per_tok;
    uint32_t head_dim;
    uint32_t embed_format, attn_format, expert_format, group_size;
    float rope_theta, rms_norm_eps, embedding_multiplier;
    float attention_multiplier, residual_multiplier, logits_scaling;

    uint64_t hot_offset, hot_size;
    uint64_t expert_dir_offset;
    uint32_t expert_dir_count, expert_dir_entry_size;
    uint64_t experts_offset;
    uint64_t tokenizer_offset, tokenizer_size;

    void *hot_base;
    size_t hot_len;

    uint16_t *embed_f16;
    float *norms_f32;
    float *routers_f32;
    float *inv_freq;

    crom_layer_t *layers;
    crom_dir_entry_t *dir;

    void *expert_pool;
    struct lru_t *lru;
    int cache_experts;
    int streaming;
    int fd;
    int threads;
    int max_seq;             /* contexto dinâmico */
    size_t expert_size;      /* tamanho em bytes de cada expert */
    int total_experts;       /* num_layers * num_local_experts */
    const char *path;        /* caminho do arquivo .cromllm */

    float *k_cache;
    float *v_cache;

    float *work1;
    float *moe_acc_buf;
    float *attn_scores;      /* buffer de scores de tamanho max_seq */

    /* buffers dinâmicos de roteamento MoE */
    float *router_logits;
    int   *router_idx;
    int   *top_experts;
    float *top_weights;
    float *top_logits;

    /* prefetch assíncrono SSD via pthread */
    int sync_io;                      /* 1 = síncrono, 0 = assíncrono (padrão) */
    struct crom_async_io_t *aio;

    /* buffers dedicados para batch prefill */
    float *batch_work;
    size_t batch_work_cap;

    long long cache_hits;
    long long cache_misses;
    long long ssd_bytes_read;
    long long ssd_stream_bytes_read;
    double ssd_io_ms;
    double attn_ms;
    double moe_ms;
    double lm_head_ms;
} crom_model_t;

typedef struct {
    int threads;
    int cache_experts;
    int steps;
    int dump_ckpt;
} crom_cfg_t;

/* ------------------------------------------------------------------ */
/* API                                                                */
/* ------------------------------------------------------------------ */

int crom_model_open(const char *path, crom_model_t *m, int cache_experts, int threads);
int crom_model_open_ctx(const char *path, crom_model_t *m, int cache_experts, int threads, int max_seq);
void crom_model_close(crom_model_t *m);
int crom_forward(crom_model_t *m, const int *input_ids, int n_ids,
                 int position, float *hidden, float *out_logits);
int crom_forward_batch(crom_model_t *m, const int *input_ids, int n_tokens,
                       int start_pos, float *out_logits);
void crom_set_sync_io(crom_model_t *m, int sync_io);
int crom_embed(crom_model_t *m, int token_id, float *hidden);
int crom_layer_attn_resid(crom_model_t *m, int li, const float *hidden_in,
                          int pos, float *hidden_out);
int crom_layer_moe_resid(crom_model_t *m, int li, const float *hidden_in,
                          int pos, float *hidden_out,
                          int *top8_out, float *top8_w_out);
void crom_rms_norm(const float *w, const float *x, int n, float eps, float *out);
void crom_stats(const crom_model_t *m, long long *hits, long long *misses,
                long long *bytes, long long *stream_bytes, double *io_ms,
                double *attn_ms, double *moe_ms, double *lm_head_ms);
void crom_reset_kv(crom_model_t *m);

/* Getters para Go/C */
int crom_vocab_size(const crom_model_t *m);
int crom_hidden_size(const crom_model_t *m);
int crom_num_experts_total(const crom_model_t *m);
int crom_max_seq(const crom_model_t *m);

/* tokenizer embutido */
const void *crom_tokenizer_data(const crom_model_t *m, size_t *out_size);
int crom_eos_id(void);  /* 0 = <|end_of_text|> */

#endif /* CROM_INFER_H */
