/* EXP-005 (T2): duelo de kernels de matvec MoE em C/AVX2.
 * Working set: 32 experts x 24 camadas (input 1024x1024 + output 512x1024)
 * = 1.207.959.552 pesos (4,5 GiB em FP32) >> L3 (16 MB).
 * 10 tokens de decode (192 matvecs/token: 8 experts x 24 camadas).
 * Compilar: gcc -O3 -march=native -fopenmp -Wall -o bench_kernels bench_kernels.c
 */
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define L 24
#define E 32
#define DIM 1024
#define FF 512
#define G 64
#define K 256
#define TOKENS 10
#define MATS (L * E)              /* 768 matrizes */
#define EIN ((size_t)DIM * DIM)   /* 1.048.576 */
#define EOUT ((size_t)FF * DIM)   /* 524.288 */
#define TOT (MATS * (EIN + EOUT)) /* 1.207.959.552 */

static float *win, *wout, *ref_mat;
static uint8_t *q4_idx, *b1_idx, *l16_idx;   /* bloco in (MATS*EIN/2) | bloco out */
static float *q4_s, *b1_s, *l16_s;           /* escalas: e/64 floats por matriz */
static float *b1_cb;                          /* [MATS][K][2] dominio normalizado */
static int8_t *l16_cb;                        /* [MATS][e/64 grupos][16] */
static size_t off_e[2][MATS], off_i[2][MATS], off_s[2][MATS];
static float *lut_pool;                       /* LUT por thread: [512][256] floats */

static uint64_t xs = 0x9E3779B97F4A7C15ull;
static inline uint64_t xorshift(void) {
    xs ^= xs << 13; xs ^= xs >> 7; xs ^= xs << 17; return xs;
}
static inline float frand(void) {
    return ((float)(xorshift() & 0xFFFF) / 65535.0f) * 0.1f - 0.05f;
}

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

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

static size_t rows_of(int m) { return m == 0 ? DIM : FF; }
static size_t elems_of(int m) { return m == 0 ? EIN : EOUT; }

static inline uint8_t *blk_i(int m, int fmt, size_t mi) {
    uint8_t *base = fmt == 0 ? q4_idx : fmt == 1 ? b1_idx : l16_idx;
    return base + (m == 0 ? 0 : MATS * EIN / 2) + off_i[m][mi];
}
static inline float *blk_s(int m, int fmt, size_t mi) {
    float *base = fmt == 0 ? q4_s : fmt == 1 ? b1_s : l16_s;
    return base + (m == 0 ? 0 : MATS * EIN / G) + off_s[m][mi];
}

/* ---------------- kernels ---------------- */

static void kv_fp32(const float *W, size_t rows, const float *x, float *y) {
    for (size_t r = 0; r < rows; r++) {
        const float *w = W + r * DIM;
        __m256 acc = _mm256_setzero_ps();
        for (size_t c = 0; c < DIM; c += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + c), _mm256_loadu_ps(x + c), acc);
        y[r] = hsum256(acc);
    }
}

static void kv_q4(const uint8_t *idx, const float *sc, size_t rows, const float *x, float *y) {
    const __m256i maskF = _mm256_set1_epi8(0x0F), v8 = _mm256_set1_epi8(8);
    for (size_t r = 0; r < rows; r++) {
        const uint8_t *ip = idx + r * (DIM / 2);
        const float *sp = sc + r * (DIM / G);
        __m256 acc = _mm256_setzero_ps();
        for (size_t g = 0; g < DIM / G; g++) {
            __m256i p = _mm256_loadu_si256((const __m256i *)(ip + g * 32));
            __m256i lo = _mm256_sub_epi8(_mm256_and_si256(p, maskF), v8);
            __m256i hi = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(p, 4), maskF), v8);
            __m256 accg = _mm256_setzero_ps(), f[4];
            const float *xg = x + g * G;
            for (int half = 0; half < 2; half++) {
                i8_to_f32_32(half ? hi : lo, f);
                for (int i = 0; i < 4; i++)
                    accg = _mm256_fmadd_ps(f[i], _mm256_loadu_ps(xg + half * 32 + i * 8), accg);
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(sp[g]), accg, acc);
        }
        y[r] = hsum256(acc);
    }
}

/* LUT[j][c] = cb[2c]*x[2j] + cb[2c+1]*x[2j+1] (cb no dominio normalizado) */
static void build_lut_into(const float *cb, const float *x, float *dst) {
    for (int j = 0; j < DIM / 2; j++) {
        float a = x[2 * j], b = x[2 * j + 1];
        float *row = dst + (size_t)j * K;
        for (int c = 0; c < K; c++)
            row[c] = cb[2 * c] * a + cb[2 * c + 1] * b;
    }
}

static void kv_b1(const uint8_t *idx, const float *sc, const float *lutp,
                  size_t rows, float *y) {
    /* endereco da LUT p/ par p, codigo cd: p*K + cd; p = g*32 + u*8 + i (i=lane) */
    const __m256i lane_off = _mm256_setr_epi32(0, K, 2*K, 3*K, 4*K, 5*K, 6*K, 7*K);
    for (size_t r = 0; r < rows; r++) {
        const uint8_t *cp = idx + r * (DIM / 2);
        const float *sp = sc + r * (DIM / G);
        __m256 acc = _mm256_setzero_ps();
        for (size_t g = 0; g < DIM / G; g++) {
            __m256 m = _mm256_set1_ps(sp[g]);
            for (int u = 0; u < 4; u++) {
                __m256i ci = _mm256_cvtepu8_epi32(
                    _mm_loadu_si128((const __m128i *)(cp + g * 32 + u * 8)));
                ci = _mm256_add_epi32(ci, lane_off);
                ci = _mm256_add_epi32(ci, _mm256_set1_epi32((int)((g * 32 + u * 8) * K)));
                __m256 v = _mm256_i32gather_ps(lutp, ci, 4);
                acc = _mm256_fmadd_ps(m, v, acc);
            }
        }
        y[r] = hsum256(acc);
    }
}

static void kv_lut16(const uint8_t *idx, const int8_t *cb, const float *sc,
                     size_t rows, const float *x, float *y) {
    (void)x;
    const __m256i maskF = _mm256_set1_epi8(0x0F);
    for (size_t r = 0; r < rows; r++) {
        const uint8_t *ip = idx + r * (DIM / 2);
        const float *sp = sc + r * (DIM / G);
        __m256 acc = _mm256_setzero_ps();
        for (size_t g = 0; g < DIM / G; g++) {
            __m256i cbv = _mm256_broadcastsi128_si256(
                _mm_loadu_si128((const __m128i *)(cb + (r * (DIM / G) + g) * 16)));
            __m256i p = _mm256_loadu_si256((const __m256i *)(ip + g * 32));
            __m256i lo = _mm256_shuffle_epi8(cbv, _mm256_and_si256(p, maskF));
            __m256i hi = _mm256_shuffle_epi8(cbv, _mm256_and_si256(_mm256_srli_epi16(p, 4), maskF));
            __m256 accg = _mm256_setzero_ps(), f[4];
            const float *xg = x + g * G;
            for (int half = 0; half < 2; half++) {
                i8_to_f32_32(half ? hi : lo, f);
                for (int i = 0; i < 4; i++)
                    accg = _mm256_fmadd_ps(f[i], _mm256_loadu_ps(xg + half * 32 + i * 8), accg);
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(sp[g]), accg, acc);
        }
        y[r] = hsum256(acc);
    }
}

/* ---------------- construcao dos formatos ---------------- */

static void build_q4(int m, size_t mi) {
    size_t rows = rows_of(m);
    float *W = m == 0 ? win + off_e[0][mi] : wout + off_e[1][mi];
    uint8_t *base = blk_i(m, 0, mi);
    float *sc = blk_s(m, 0, mi);
    for (size_t r = 0; r < rows; r++) {
        const float *w = W + r * DIM;
        for (size_t g = 0; g < DIM / G; g++) {
            float s = 0.0f;
            for (size_t c = 0; c < G; c++) {
                float a = w[g * G + c]; if (a < 0) a = -a; if (a > s) s = a;
            }
            s /= 7.0f;
            sc[r * (DIM / G) + g] = s;
            for (size_t c = 0; c < G / 2; c++) {
                float v0 = w[g * G + c] / s, v1 = w[g * G + c + 32] / s;
                int c0 = (int)(v0 + (v0 >= 0 ? 0.5f : -0.5f));
                int c1 = (int)(v1 + (v1 >= 0 ? 0.5f : -0.5f));
                if (c0 > 7) c0 = 7;
                if (c0 < -7) c0 = -7;
                if (c1 > 7) c1 = 7;
                if (c1 < -7) c1 = -7;
                base[r * (DIM / 2) + g * 32 + c] =
                    (uint8_t)(((c0 + 8) & 0xF) | (((c1 + 8) & 0xF) << 4));
            }
        }
    }
}

static void build_b1(int m, size_t mi) {
    size_t rows = rows_of(m), np = elems_of(m) / 2;
    float *W = m == 0 ? win + off_e[0][mi] : wout + off_e[1][mi];
    uint8_t *base = blk_i(m, 1, mi);
    float *sc = blk_s(m, 1, mi);
    float *cb = b1_cb + (size_t)((m == 0 ? 0 : MATS) + mi) * 2 * K;

    enum { NS = 10000 };
    float (*samp)[2] = malloc((size_t)NS * 2 * sizeof(float));
    size_t stride = np / NS; if (stride == 0) stride = 1;
    size_t ns = 0;
    for (size_t p = 0; p < np && ns < NS; p += stride) {
        size_t r = p / (DIM / 2), j = p % (DIM / 2), g = (2 * j) / G;
        float s = 0.0f;
        for (size_t c = 0; c < G; c++) {
            float a = W[r * DIM + g * G + c]; if (a < 0) a = -a; if (a > s) s = a;
        }
        if (s < 1e-12f) s = 1e-12f;
        samp[ns][0] = W[r * DIM + 2 * j] / s;
        samp[ns][1] = W[r * DIM + 2 * j + 1] / s;
        ns++;
    }
    float *C = malloc((size_t)K * 2 * sizeof(float));
    int *as = malloc(ns * sizeof(int));
    for (int i = 0; i < K; i++) { C[2*i] = samp[(i * ns) / K][0]; C[2*i+1] = samp[(i * ns) / K][1]; }
    for (int it = 0; it < 3; it++) {
        for (size_t i = 0; i < ns; i++) {
            int best = 0; float bd = 1e30f;
            for (int k = 0; k < K; k++) {
                float dx = samp[i][0]-C[2*k], dy = samp[i][1]-C[2*k+1];
                float d = dx*dx + dy*dy;
                if (d < bd) { bd = d; best = k; }
            }
            as[i] = best;
        }
        float sum[K][2]; int cnt[K];
        memset(sum, 0, sizeof(sum)); memset(cnt, 0, sizeof(cnt));
        for (size_t i = 0; i < ns; i++) { sum[as[i]][0]+=samp[i][0]; sum[as[i]][1]+=samp[i][1]; cnt[as[i]]++; }
        for (int k = 0; k < K; k++) if (cnt[k]) { C[2*k]=sum[k][0]/cnt[k]; C[2*k+1]=sum[k][1]/cnt[k]; }
    }
    free(as); free(samp);
    memcpy(cb, C, (size_t)K * 2 * sizeof(float));

#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rows; r++) {
        const float *w = W + r * DIM;
        for (size_t g = 0; g < DIM / G; g++) {
            float s = 0.0f;
            for (size_t c = 0; c < G; c++) { float a = w[g*G+c]; if (a<0) a=-a; if (a>s) s=a; }
            if (s < 1e-12f) s = 1e-12f;
            sc[r*(DIM/G)+g] = s;
            for (size_t c = 0; c < G/2; c++) {
                float a = w[g*G+2*c]/s, b = w[g*G+2*c+1]/s;
                int best = 0; float bd = 1e30f;
                for (int k = 0; k < K; k++) {
                    float dx = a-C[2*k], dy = b-C[2*k+1], d = dx*dx+dy*dy;
                    if (d < bd) { bd = d; best = k; }
                }
                base[r*(DIM/2) + g*(G/2) + c] = (uint8_t)best;
            }
        }
    }
    free(C);
}

static void build_lut16(int m, size_t mi) {
    size_t rows = rows_of(m);
    float *W = m == 0 ? win + off_e[0][mi] : wout + off_e[1][mi];
    uint8_t *base = blk_i(m, 2, mi);
    float *sc = blk_s(m, 2, mi);
    int8_t *cb = l16_cb + (m == 0 ? 0 : MATS * EIN / G * 16) + off_s[m][mi] * 16;

    for (size_t r = 0; r < rows; r++) {
        const float *w = W + r * DIM;
        for (size_t g = 0; g < DIM / G; g++) {
            const float *wg = w + g * G;
            float v[G];
            memcpy(v, wg, G * sizeof(float));
            for (int i = G - 1; i > 0; i--)
                for (int j2 = 0; j2 < i; j2++)
                    if (v[j2] > v[j2+1]) { float t = v[j2]; v[j2] = v[j2+1]; v[j2+1] = t; }
            float s = v[G-1] < 0 ? -v[G-1] : v[G-1];
            if (-v[0] > s) s = -v[0];
            if (s < 1e-12f) s = 1e-12f;
            s /= 127.0f;
            sc[r*(DIM/G)+g] = s;
            int8_t *cg = cb + (r*(DIM/G)+g)*16;
            float cent[16];
            for (int k = 0; k < 16; k++) {
                float sum = 0;
                for (int q = 0; q < 4; q++) sum += v[k*4+q];
                cent[k] = sum / 4.0f / s;
                if (cent[k] > 127) cent[k] = 127;
                if (cent[k] < -127) cent[k] = -127;
                cg[k] = (int8_t)lrintf(cent[k]);
            }
            for (size_t c = 0; c < G / 2; c++) {
                float q0 = wg[c] / s, q1 = wg[c + 32] / s;
                int b0 = 0, b1 = 0; float bd0 = 1e30f, bd1 = 1e30f;
                for (int k = 0; k < 16; k++) {
                    float d0 = q0 - cent[k]; if (d0 < 0) d0 = -d0;
                    if (d0 < bd0) { bd0 = d0; b0 = k; }
                    float d1 = q1 - cent[k]; if (d1 < 0) d1 = -d1;
                    if (d1 < bd1) { bd1 = d1; b1 = k; }
                }
                base[r*(DIM/2) + g*32 + c] = (uint8_t)(b0 | (b1 << 4));
            }
        }
    }
}

/* ---------------- validacao ---------------- */

static void validate(const float *x) {
    printf("== validacao (matriz ref: layer0/expert0 input 1024x1024) ==\n");
    float yk[DIM], ys[DIM];
    double nW2 = 0;
    for (size_t i = 0; i < EIN; i++) nW2 += (double)ref_mat[i] * ref_mat[i];

    /* --- Q4 --- */
    double e2 = 0, q2 = 0, ny = 0;
    kv_q4(blk_i(0, 0, 0), blk_s(0, 0, 0), DIM, x, yk);
    for (size_t r = 0; r < DIM; r++) {
        const float *w = ref_mat + r * DIM;
        double acc = 0;
        for (size_t g = 0; g < DIM / G; g++) {
            float s = q4_s[r*(DIM/G)+g], ag = 0;
            for (size_t c = 0; c < G; c++) {
                uint8_t byte = q4_idx[r*(DIM/2) + g*32 + (c & 31)];
                int code = (c < 32 ? (byte & 0xF) : (byte >> 4)) - 8;
                ag += (double)code * x[g*G+c];
                q2 += ((double)s*code - w[g*G+c]) * ((double)s*code - w[g*G+c]);
            }
            acc += (double)s * ag;
        }
        ys[r] = (float)acc;
        e2 += ((double)yk[r]-ys[r]) * ((double)yk[r]-ys[r]);
        ny += (double)ys[r] * ys[r];
    }
    printf("  Q4 : simd-vs-scalar max|d|=%.3e rel=%.3e | dequant-vs-fp32 rel=%.4f\n",
           sqrt(e2), sqrt(e2/ny), sqrt(q2/nW2));

    /* --- B1 --- */
    float *lut0 = lut_pool;
    float *cb0 = b1_cb;
    e2 = q2 = ny = 0;
    build_lut_into(cb0, x, lut0);
    kv_b1(blk_i(0, 1, 0), blk_s(0, 1, 0), lut0, DIM, yk);
    for (size_t r = 0; r < DIM; r++) {
        const float *w = ref_mat + r * DIM;
        double acc = 0;
        for (size_t g = 0; g < DIM / G; g++) {
            float s = b1_s[r*(DIM/G)+g], ag = 0;
            for (size_t c = 0; c < G/2; c++) {
                int code = b1_idx[r*(DIM/2) + g*(G/2) + c];
                double p = (double)cb0[2*code]*x[g*G+2*c] + (double)cb0[2*code+1]*x[g*G+2*c+1];
                ag += p;
                q2 += ((double)s*cb0[2*code] - w[g*G+2*c]) * ((double)s*cb0[2*code] - w[g*G+2*c]);
                q2 += ((double)s*cb0[2*code+1] - w[g*G+2*c+1]) * ((double)s*cb0[2*code+1] - w[g*G+2*c+1]);
            }
            acc += (double)s * ag;
        }
        ys[r] = (float)acc;
        e2 += ((double)yk[r]-ys[r]) * ((double)yk[r]-ys[r]);
        ny += (double)ys[r] * ys[r];
    }
    printf("  B1 : simd-vs-scalar max|d|=%.3e rel=%.3e | dequant-vs-fp32 rel=%.4f\n",
           sqrt(e2), sqrt(e2/ny), sqrt(q2/nW2));

    /* --- LUT16 --- */
    int8_t *cb16 = l16_cb + off_s[0][0] * 16;
    e2 = q2 = ny = 0;
    kv_lut16(blk_i(0, 2, 0), cb16, blk_s(0, 2, 0), DIM, x, yk);
    for (size_t r = 0; r < DIM; r++) {
        const float *w = ref_mat + r * DIM;
        double acc = 0;
        for (size_t g = 0; g < DIM / G; g++) {
            float s = l16_s[r*(DIM/G)+g], ag = 0;
            const int8_t *cg = cb16 + (r*(DIM/G)+g)*16;
            for (size_t c = 0; c < G; c++) {
                uint8_t byte = l16_idx[r*(DIM/2) + g*32 + (c & 31)];
                int code = c < 32 ? (byte & 0xF) : (byte >> 4);
                ag += (double)cg[code] * x[g*G+c];
                q2 += ((double)s*cg[code] - w[g*G+c]) * ((double)s*cg[code] - w[g*G+c]);
            }
            acc += (double)s * ag;
        }
        ys[r] = (float)acc;
        e2 += ((double)yk[r]-ys[r]) * ((double)yk[r]-ys[r]);
        ny += (double)ys[r] * ys[r];
    }
    printf("  LUT16: simd-vs-scalar max|d|=%.3e rel=%.3e | dequant-vs-fp32 rel=%.4f\n",
           sqrt(e2), sqrt(e2/ny), sqrt(q2/nW2));
}

/* ---------------- bench ---------------- */

static int sel[TOKENS][L][8];

static double token_sum(int t, int mode) { /* mode: 0 fp32, 1 q4, 2 b1, 3 lut16 */
    float x[DIM];
    xs = 1000 + (uint64_t)t;
    for (int i = 0; i < DIM; i++) x[i] = frand();
    double cloc = 0;
#pragma omp parallel for reduction(+:cloc) schedule(dynamic, 8)
    for (int job = 0; job < L * 8; job++) {
        int l = job / 8, k = job % 8, e = sel[t][l][k];
        float y1[DIM], y2[FF];
        if (mode == 0) {
            kv_fp32(win + off_e[0][l * E + e], DIM, x, y1);
            kv_fp32(wout + off_e[1][l * E + e], FF, y1, y2);
        } else if (mode == 1) {
            kv_q4(blk_i(0, 0, l * E + e), blk_s(0, 0, l * E + e), DIM, x, y1);
            kv_q4(blk_i(1, 0, l * E + e), blk_s(1, 0, l * E + e), FF, y1, y2);
        } else if (mode == 2) {
            int tid = omp_get_thread_num();
            float *mylut = lut_pool + (size_t)tid * (DIM / 2) * K;
            build_lut_into(b1_cb + (size_t)(l * E + e) * 2 * K, x, mylut);
            kv_b1(blk_i(0, 1, l * E + e), blk_s(0, 1, l * E + e), mylut, DIM, y1);
            build_lut_into(b1_cb + (size_t)(MATS + l * E + e) * 2 * K, y1, mylut);
            kv_b1(blk_i(1, 1, l * E + e), blk_s(1, 1, l * E + e), mylut, FF, y2);
        } else {
            kv_lut16(blk_i(0, 2, l * E + e), l16_cb + off_s[0][l * E + e] * 16,
                     blk_s(0, 2, l * E + e), DIM, x, y1);
            kv_lut16(blk_i(1, 2, l * E + e),
                     l16_cb + (MATS * EIN / G + off_s[1][l * E + e]) * 16,
                     blk_s(1, 2, l * E + e), FF, y1, y2);
        }
        for (int i = 0; i < FF; i++) cloc += y2[i];
    }
    return cloc;
}

static void bench(int mode, const char *name, double bytes_tok) {
    double chk = token_sum(0, mode);  /* warmup */
    double t0 = now_s();
    for (int t = 0; t < TOKENS; t++) chk += token_sum(t, mode);
    double dt = now_s() - t0;
    printf("  %-14s: %7.2f ms/token | %6.2f tok/s | %6.1f GB/s | chk=%.4f\n",
           name, dt / TOKENS * 1000, TOKENS / dt, bytes_tok * TOKENS / dt / 1e9, chk);
}

int main(int argc, char **argv) {
    int quick = argc > 1;  /* so matriz 0 + validacao, sem bench */
    size_t mats_build = quick ? 1 : MATS;
    printf("working set FP32: %.2f GiB (%zu pesos)\n", TOT * 4.0 / (1 << 30), TOT);
    win = calloc(MATS * EIN, 4);
    wout = calloc(MATS * EOUT, 4);
    ref_mat = malloc(EIN * 4);
    lut_pool = aligned_alloc(64, (size_t)omp_get_max_threads() * (DIM / 2) * K * 4);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < TOT; i++) {
        size_t split = MATS * EIN;
        if (i < split) win[i] = frand(); else wout[i - split] = frand();
    }
    memcpy(ref_mat, win, EIN * 4);
    printf("matrizes preenchidas\n");

    for (int mm = 0; mm < 2; mm++) {
        size_t acci = 0, accs = 0;
        for (size_t mi = 0; mi < MATS; mi++) {
            off_e[mm][mi] = mm == 0 ? mi * EIN : mi * EOUT;
            off_i[mm][mi] = acci; off_s[mm][mi] = accs;
            acci += elems_of(mm) / 2; accs += elems_of(mm) / G;
        }
    }

    q4_idx = malloc(MATS * (EIN + EOUT) / 2); q4_s = malloc(MATS * (EIN + EOUT) / G * 4);
    b1_idx = malloc(MATS * (EIN + EOUT) / 2); b1_s = malloc(MATS * (EIN + EOUT) / G * 4);
    b1_cb = malloc((size_t)2 * MATS * 2 * K * 4);
    l16_idx = malloc(MATS * (EIN + EOUT) / 2); l16_s = malloc(MATS * (EIN + EOUT) / G * 4);
    l16_cb = malloc(MATS * (EIN + EOUT) / G * 16);
    if (!q4_idx || !q4_s || !b1_idx || !b1_s || !b1_cb || !l16_idx || !l16_s || !l16_cb) {
    }

    printf("construindo Q4...\n");
#pragma omp parallel for schedule(static)
    for (size_t mi = 0; mi < mats_build; mi++) { build_q4(0, mi); build_q4(1, mi); }

    printf("construindo B1 (kmeans-lite K=256, d=2)...\n");
#pragma omp parallel for schedule(static)
    for (size_t mi = 0; mi < mats_build; mi++) { build_b1(0, mi); build_b1(1, mi); }

    printf("construindo LUT16...\n");
#pragma omp parallel for schedule(static)
    for (size_t mi = 0; mi < mats_build; mi++) { build_lut16(0, mi); build_lut16(1, mi); }

    float xv[DIM];
    xs = 42;
    for (int i = 0; i < DIM; i++) xv[i] = frand();
    validate(xv);
    if (quick) { printf("[quick] fim\n"); return 0; }

    printf("formatos compactos residentes (~1,9 GiB); FP32 permanece p/ bench base\n\n");

    for (int t = 0; t < TOKENS; t++)
        for (int l = 0; l < L; l++) {
            int s = (t * 73 + l * 131) & 31;
            for (int k = 0; k < 8; k++) sel[t][l][k] = (s + 7 * k) & 31;
        }

    double bytes_tok[4] = {
        192.0 * (EIN + EOUT) * 4,
        192.0 * ((EIN + EOUT) / 2 + (EIN + EOUT) / 64.0 * 4),
        192.0 * ((EIN + EOUT) / 2 + (EIN + EOUT) / 64.0 * 4 + 2 * K * 4),
        192.0 * ((EIN + EOUT) / 2 + (EIN + EOUT) / 64.0 * (4 + 16)),
    };

    for (int cfg = 0; cfg < 3; cfg++) {
        int nt = cfg == 0 ? 1 : cfg == 1 ? 6 : 12;
        omp_set_num_threads(nt);
        printf("== OMP_NUM_THREADS=%d ==\n", nt);
        bench(0, "FP32_AVX2", bytes_tok[0]);
        bench(1, "Q4_g64_AVX2", bytes_tok[1]);
        bench(2, "B1_LUT_d2_K256", bytes_tok[2]);
        bench(3, "LUT16_VPSHUFB", bytes_tok[3]);
        printf("\n");
    }

    free(q4_idx); free(q4_s); free(b1_idx); free(b1_s); free(b1_cb);
    free(l16_idx); free(l16_s); free(l16_cb); free(ref_mat); free(lut_pool);
    free(win); free(wout);
    printf("fim\n");
    return 0;
}
