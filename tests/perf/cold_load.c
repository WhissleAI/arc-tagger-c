/* Load with the weights not in the file cache, without needing root: each run writes fresh copies of the model files uncached
 * (F_NOCACHE), checks with mincore that they really are not resident, and times arctag_create and the first short text.
 * macOS only. usage: cold_load MODEL_DIR TMP_DIR [RUNS] */
#include "arctag.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static void copy_uncached(const char *src, const char *dst) {
    int in = open(src, O_RDONLY), out = open(dst, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (in < 0 || out < 0) { perror(src); exit(1); }
    fcntl(out, F_NOCACHE, 1); fcntl(in, F_NOCACHE, 1);
    size_t cap = 1 << 20; char *b = NULL; if (posix_memalign((void **)&b, 4096, cap)) exit(1);
    ssize_t n; while ((n = read(in, b, cap)) > 0) if (write(out, b, (size_t)n) != n) { perror(dst); exit(1); }
    fsync(out); close(in); close(out); free(b);
}
static double resident_fraction(const char *path) {
    int fd = open(path, O_RDONLY); struct stat st; fstat(fd, &st);
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); size_t pg = (size_t)getpagesize(), np = ((size_t)st.st_size + pg - 1) / pg;
    char *vec = malloc(np); long in = 0;
    if (!mincore(p, (size_t)st.st_size, vec)) for (size_t i = 0; i < np; i++) in += vec[i] & 1;
    free(vec); munmap(p, (size_t)st.st_size); close(fd);
    return (double)in / (double)np;
}
int main(int c, char **v) {
    if (c < 3) { fprintf(stderr, "usage: %s MODEL_DIR TMP_DIR [RUNS]\n", v[0]); return 2; }
    int runs = c > 3 ? atoi(v[3]) : 5; char a[1024], b[1024], err[512];
    mkdir(v[2], 0755);
    const char *files[] = {"arctag.atw", "arctag_meta.json", "tokenizer.json"};
    for (int r = 0; r < runs; r++) {
        for (int f = 0; f < 3; f++) { snprintf(a, sizeof a, "%s/%s", v[1], files[f]); snprintf(b, sizeof b, "%s/%s", v[2], files[f]); copy_uncached(a, b); }
        snprintf(b, sizeof b, "%s/arctag.atw", v[2]); double before = resident_fraction(b);
        double t = now_ms(); arctag_t *h = arctag_create(v[2], err, sizeof err); double load = now_ms() - t;
        if (!h) { fprintf(stderr, "%s\n", err); return 1; }
        t = now_ms(); char *o = arctag_tag_text(h, "Can you call me back tomorrow?"); double first = now_ms() - t; arctag_free_str(o);
        double after = resident_fraction(b);
        printf("run %d: weights in file cache before %.1f%%, after %.1f%%; load %.1f ms, first short text %.1f ms\n", r, 100 * before, 100 * after, load, first);
        arctag_destroy(h);
    }
    return 0;
}
