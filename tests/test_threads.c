/* Many handles on many threads: results must equal a single-threaded run, byte for byte, and ThreadSanitizer must stay quiet
 * (scripts/test_native.sh builds this with -fsanitize=thread). Also one model shared by several atn states, as atn.h allows, and
 * concurrent create/destroy. usage: test_threads MODEL_DIR THREADS ROUNDS */
#include "arctag.h"
#include "atn.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *TEXT = "Can you call me back at 415 555 0199 tomorrow? Priya sends the final numbers by Friday. "
    "Hi, this is Maria Lopez calling about my order. The Lakers are up 98 to 95 with two minutes left. Okay team, let's start.";
static const char *dir; static int rounds;
static char *want_tag, *want_stream;

/* word by word with a revision every 7 words, as an ASR would */
static char *run_stream(arctag_t *h, int words) {
    char *t = strdup(TEXT), *w[64]; int nw = 0;
    for (char *x = strtok(t, " "); x; x = strtok(NULL, " ")) w[nw++] = x;
    char js[256]; char *last = NULL;
    for (int i = 0; i < words; i++) {
        snprintf(js, sizeof js, "[\"%s\"]", w[i % nw]);
        if (arctag_push_words(h, js)) { fprintf(stderr, "push: %s\n", arctag_last_error(h)); exit(1); }
        if (i % 7 == 6) { snprintf(js, sizeof js, "{\"from\":%d,\"words\":[\"%s\",\"%s\"]}", i - 1, w[(i - 1) % nw], w[i % nw]); if (arctag_push_words(h, js)) { fprintf(stderr, "revise: %s\n", arctag_last_error(h)); exit(1); } }
        free(last); last = arctag_poll(h);
        if (!last) { fprintf(stderr, "poll: %s\n", arctag_last_error(h)); exit(1); }
    }
    free(t);
    return last;
}
static int diffs = 0; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static void *worker(void *arg) {
    (void)arg; char err[512];
    arctag_t *h = arctag_create(dir, err, sizeof err);
    if (!h) { fprintf(stderr, "create: %s\n", err); exit(1); }
    for (int r = 0; r < rounds; r++) {
        char *a = arctag_tag_text(h, TEXT);
        arctag_reset(h);
        char *b = run_stream(h, 60);
        arctag_reset(h);
        pthread_mutex_lock(&mu);
        if (!a || strcmp(a, want_tag)) diffs++;
        if (!b || strcmp(b, want_stream)) diffs++;
        pthread_mutex_unlock(&mu);
        arctag_free_str(a); arctag_free_str(b);
    }
    arctag_destroy(h);
    return NULL;
}
static void *churn(void *arg) {
    (void)arg; char err[512];
    for (int r = 0; r < 3; r++) { arctag_t *h = arctag_create(dir, err, sizeof err); if (!h) { fprintf(stderr, "create: %s\n", err); exit(1); } arctag_free_str(arctag_tag_text(h, "short text.")); arctag_destroy(h); }
    return NULL;
}
static atn_model *shared; static float *want_rows;
static void *shared_worker(void *arg) {
    (void)arg; int tot = atn_out_total(shared); atn *a = atn_new(shared, 0, 256, 160);
    atn_word ws[40]; memset(ws, 0, sizeof ws);
    for (int i = 0; i < 40; i++) { ws[i].npc = 2; ws[i].pcs[0] = 5 + i; ws[i].pcs[1] = 300 + i; ws[i].feats[2] = 1; ws[i].ust = i % 9 == 0; }
    float *o = malloc(sizeof(float) * 40 * (size_t)tot);
    for (int r = 0; r < rounds; r++) {
        atn_reset(a); atn_steps(a, ws, 40, o);
        pthread_mutex_lock(&mu);
        if (want_rows && memcmp(o, want_rows, sizeof(float) * 40 * (size_t)tot)) diffs++;
        pthread_mutex_unlock(&mu);
    }
    free(o); atn_free(a);
    return NULL;
}
int main(int c, char **v) {
    if (c < 4) { fprintf(stderr, "usage: %s MODEL_DIR THREADS ROUNDS\n", v[0]); return 2; }
    dir = v[1]; int nt = atoi(v[2]); rounds = atoi(v[3]);
    char err[512], path[1024]; arctag_t *h = arctag_create(dir, err, sizeof err);
    if (!h) { fprintf(stderr, "create: %s\n", err); return 1; }
    want_tag = arctag_tag_text(h, TEXT); arctag_reset(h); want_stream = run_stream(h, 60); arctag_destroy(h);
    pthread_t *t = malloc(sizeof(pthread_t) * (size_t)(2 * nt + 4));
    int n = 0;
    for (int i = 0; i < nt; i++) pthread_create(&t[n++], NULL, worker, NULL);
    for (int i = 0; i < 2; i++) pthread_create(&t[n++], NULL, churn, NULL);
    for (int i = 0; i < n; i++) pthread_join(t[i], NULL);
    int streams = nt * rounds;
    snprintf(path, sizeof path, "%s/arctag.atw", dir);
    shared = atn_open(path, err, sizeof err);
    if (!shared) { fprintf(stderr, "open: %s\n", err); return 1; }
    /* the reference rows, one thread; the batch path is deterministic */
    { int tot = atn_out_total(shared); atn *a = atn_new(shared, 0, 256, 160); atn_word ws[40]; memset(ws, 0, sizeof ws);
      for (int i = 0; i < 40; i++) { ws[i].npc = 2; ws[i].pcs[0] = 5 + i; ws[i].pcs[1] = 300 + i; ws[i].feats[2] = 1; ws[i].ust = i % 9 == 0; }
      want_rows = malloc(sizeof(float) * 40 * (size_t)tot); atn_steps(a, ws, 40, want_rows); atn_free(a); }
    n = 0;
    for (int i = 0; i < nt; i++) pthread_create(&t[n++], NULL, shared_worker, NULL);
    for (int i = 0; i < n; i++) pthread_join(t[i], NULL);
    atn_close(shared); free(want_rows); free(t); arctag_free_str(want_tag); arctag_free_str(want_stream);
    printf("threads: %d threads x %d rounds (%d tag_text, %d streams of 60 words with revisions), 2 threads churning create/destroy, 1 shared model x %d states: %d differences from the single-threaded output\n", nt, rounds, streams, streams, nt, diffs);
    return diffs ? 1 : 0;
}
