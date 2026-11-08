/* An int8 model (tools/quantize_model.py) must be about as fast as fp32 on short texts. If ORT stops folding the
 * DequantizeLinear nodes at load, they re-run on every call and short texts get about 3x slower.
 * usage: test_int8_speed FP32_MODEL_DIR INT8_MODEL_DIR. Timing is relative, so it tolerates a slow machine. */
#include "arctag.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ROUNDS 5
#define RUNS 40
static const char *TEXT = "Can we move the budget review to Thursday?";

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

static double median_ms(arctag_t *h) {
    double ts[RUNS];
    for (int i = 0; i < RUNS; i++) {
        double t = now_ms(); char *o = arctag_tag_text(h, TEXT); ts[i] = now_ms() - t;
        if (!o) { fprintf(stderr, "tag: %s\n", arctag_last_error(h)); exit(1); }
        arctag_free_str(o);
    }
    qsort(ts, RUNS, sizeof(double), cmp);
    return ts[RUNS / 2];
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s FP32_MODEL_DIR INT8_MODEL_DIR\n", argv[0]); return 2; }
    char err[512]; arctag_t *h[2];
    for (int k = 0; k < 2; k++) {
        if (!(h[k] = arctag_create(argv[k + 1], err, sizeof err))) { fprintf(stderr, "%s: %s\n", argv[k + 1], err); return 2; }
        arctag_set_option(h[k], "threads", "1");
    }
    double best[2] = {1e9, 1e9};
    for (int r = 0; r < ROUNDS; r++)                 /* alternate and keep the best round: load spikes only slow a run down */
        for (int k = 0; k < 2; k++) { double m = median_ms(h[k]); if (m < best[k]) best[k] = m; }
    arctag_destroy(h[0]); arctag_destroy(h[1]);
    int ok = best[1] <= 1.5 * best[0] + 1.0;
    printf("int8_speed: fp32 %.2f ms, int8 %.2f ms: %s\n", best[0], best[1], ok ? "ok" : "int8 too slow, DequantizeLinear not folded?");
    return ok ? 0 : 1;
}
