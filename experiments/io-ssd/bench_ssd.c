/* EXP-001: benchmark fisico do SSD SATA (KA-SSD-512).
 * Cria tmp_ssd_bench.bin (256 MiB), mede leituras sequenciais e aleatorias
 * via pread (O_DIRECT se possivel) e apaga o arquivo ao final.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define PATH "tmp_ssd_bench.bin"
#define FILE_SIZE (256u << 20)
#define CAP (64u << 20)
static const size_t blocks[] = {4u << 10, 64u << 10, 256u << 10, 1u << 20, 4u << 20};
static const char *bnames[] = {"4 KiB", "64 KiB", "256 KiB", "1 MiB", "4 MiB"};

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static void run(int fd, size_t bs, int rnd) {
    char *buf = aligned_alloc(4096, bs);
    if (!buf) { fprintf(stderr, "aligned_alloc falhou\n"); exit(1); }
    size_t limit = rnd ? (CAP / bs < 256 ? CAP / bs : 256) : (CAP / bs);
    size_t count = 0;
    double t0 = now_s();
    while (count < limit) {
        off_t off = rnd ? (off_t)(((unsigned long long)random() % (FILE_SIZE / bs)) * bs)
                        : (off_t)(count * bs);
        ssize_t r = pread(fd, buf, bs, off);
        if (r != (ssize_t)bs) { fprintf(stderr, "pread falhou (%zd, errno=%d)\n", r, errno); exit(1); }
        count++;
    }
    double dt = now_s() - t0;
    double mbs = (double)(count * bs) / dt / (1024.0 * 1024.0);
    printf("  %-13s %-9s | %8.1f MB/s | lat media %8.3f ms | %10.1f IOPS\n",
           bnames[(size_t)(bs == blocks[0] ? 0 : bs == blocks[1] ? 1 : bs == blocks[2] ? 2 : bs == blocks[3] ? 3 : 4)],
           rnd ? "aleatoria" : "sequencial", mbs, dt / count * 1000.0, count / dt);
    free(buf);
}

int main(void) {
    srandom(12345);
    char *wbuf = aligned_alloc(4096, 1u << 20);
    if (!wbuf) return 1;
    for (size_t i = 0; i < (1u << 20); i++) wbuf[i] = (char)rand();

    int fd = open(PATH, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("criar arquivo"); return 1; }
    for (size_t i = 0; i < FILE_SIZE; i += (1u << 20)) {
        if (write(fd, wbuf, 1u << 20) != (1u << 20)) { perror("write"); return 1; }
    }
    if (fsync(fd) != 0) { perror("fsync"); return 1; }
    close(fd);
    printf("arquivo de teste criado: %u MiB\n", FILE_SIZE >> 20);

    int direct = 1;
    fd = open(PATH, O_RDONLY | O_DIRECT);
    if (fd < 0 && (errno == EINVAL || errno == EOPNOTSUPP)) {
        fprintf(stderr, "O_DIRECT nao suportado (%s); fallback para page cache + posix_fadvise(DONTNEED)\n",
                strerror(errno));
        direct = 0;
        fd = open(PATH, O_RDONLY);
        if (fd < 0) { perror("abrir arquivo"); return 1; }
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    } else if (fd < 0) {
        perror("abrir O_DIRECT");
        return 1;
    }
    printf("modo: %s\n\n", direct ? "O_DIRECT" : "page cache + fadvise");
    printf("  bloco        modo      |   MB/s    | lat media        | IOPS\n");

    for (size_t i = 0; i < sizeof(blocks) / sizeof(blocks[0]); i++) {
        run(fd, blocks[i], 0);
        run(fd, blocks[i], 1);
        if (!direct) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    }
    close(fd);
    if (unlink(PATH) != 0) perror("unlink");
    free(wbuf);
    printf("\narquivo temporario removido.\n");
    return 0;
}
