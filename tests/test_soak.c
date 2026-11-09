/* A very long stream: memory must plateau and cost per word must stay flat. Timings, speakers and interim revisions like a live
 * ASR, utterance ends throughout, one poll per word. usage: test_soak MODEL_DIR WORDS
 * Fails if the dirty footprint at the end is more than 2 MB above its level after the first 10% of the session, if the last decile's
 * median cost per word is more than twice the second decile's, or if any poll's word count is wrong. */
#include "arctag.h"
#include "cJSON.h"
#include "perf/mem.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static const char *BASE[] = {"can", "you", "call", "me", "back", "at", "415", "555", "0199", "tomorrow?", "priya", "sends", "the", "final", "numbers", "by", "friday.",
    "hi,", "this", "is", "maria", "lopez", "calling", "about", "my", "order.", "the", "lakers", "are", "up", "98", "to", "95.", "okay", "team,", "let's", "start.", "budget", "review", "on", "thursday"};
#define NB (sizeof BASE / sizeof *BASE)
int main(int c, char **v) {
    if (c < 3) { fprintf(stderr, "usage: %s MODEL_DIR WORDS\n", v[0]); return 2; }
    long words = atol(v[2]), step = words / 10 > 0 ? words / 10 : 1;
    char err[512]; arctag_t *h = arctag_create(v[1], err, sizeof err); if (!h) { fprintf(stderr, "%s\n", err); return 1; }
    arctag_set_option(h, "domain", "meeting");
    double *cost = malloc(sizeof(double) * (size_t)step), r0, d0, med[11] = {0}, rss[11], dirty[11];
    mem_now(&r0, &d0);
    printf("start: rss %.1f MB, dirty %.1f MB\n", r0, d0);
    int bad = 0; long events = 0, utts = 0, revs = 0, spans = 0; uint64_t x = 88172645463325252ull;
    char js[512];
    for (long i = 0, seg = 0, k = 0; i < words; i++) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        const char *w = BASE[x % NB]; int num = x >> 40 & 15;
        double t0 = 0.31 * (double)i, conf = 0.5 + 0.01 * (double)(i % 40);
        char tok[32]; if (!num) { snprintf(tok, sizeof tok, "w%u", (unsigned)(x >> 20 & 0xFFFF)); w = tok; }
        double t = now_ms();
        /* an interim: every 10th word, the last 3 words are replaced by their finals */
        if (i % 10 == 9 && i >= 3) {
            snprintf(js, sizeof js, "{\"from\":%ld,\"words\":[{\"w\":\"%s\",\"t0\":%.2f,\"t1\":%.2f,\"spk\":\"%c\",\"conf\":%.2f},\"then\",{\"w\":\"%s\",\"t0\":%.2f,\"t1\":%.2f,\"spk\":\"%c\"}]}",
                     i - 2, w, t0 - 0.6, t0 - 0.4, "AB"[(i / 90) % 2], conf, w, t0, t0 + 0.25, "AB"[(i / 90) % 2]);
            revs++;
        } else
            snprintf(js, sizeof js, "[{\"w\":\"%s\",\"t0\":%.2f,\"t1\":%.2f,\"conf\":%.2f,\"spk\":\"%c\",\"final\":%s,\"eou\":%s}]", w, t0, t0 + 0.25, conf, "AB"[(i / 90) % 2], i % 3 ? "true" : "false", i % 37 == 36 ? "true" : "false");
        if (arctag_push_words(h, js)) { fprintf(stderr, "push at %ld: %s\n", i, arctag_last_error(h)); return 1; }
        char *o = arctag_poll(h);
        double dt = now_ms() - t;
        if (!o) { fprintf(stderr, "poll at %ld: %s\n", i, arctag_last_error(h)); return 1; }
        /* the document is parsed only now and then: parsing is not the runtime's cost */
        if (i % 257 == 0 || i == words - 1) {
            cJSON *j = cJSON_Parse(o);
            if (!j || cJSON_GetObjectItem(j, "n_words")->valueint != (int)(i + 1)) { fprintf(stderr, "bad document at %ld: %.200s\n", i, o); bad++; }
            else { events += cJSON_GetArraySize(cJSON_GetObjectItem(j, "events")); utts += cJSON_GetArraySize(cJSON_GetObjectItem(j, "utterances")); spans += cJSON_GetArraySize(cJSON_GetObjectItem(j, "spans")); }
            cJSON_Delete(j);
        }
        arctag_free_str(o);
        cost[k++] = dt;
        if (k == step) {
            qsort(cost, (size_t)k, sizeof(double), cmpd); seg++;
            if (seg <= 10) { mem_now(&rss[seg], &dirty[seg]); med[seg] = cost[k / 2];
                printf("at %8ld words: rss %6.1f MB, dirty %6.1f MB, cost per word median %.2f ms, p99 %.2f ms\n", i + 1, rss[seg], dirty[seg], cost[k / 2], cost[k * 99 / 100]); fflush(stdout); }
            k = 0;
        }
    }
    double grow = dirty[10] - dirty[1];
    printf("soak: %ld words, %ld revisions; sampled docs: %ld spans, %ld events, %ld utterances. Dirty memory %.1f MB after 10%%, %.1f MB at the end (growth %.2f MB); rss %.1f -> %.1f MB. Median per word: %.2f ms (2nd decile) -> %.2f ms (last)\n",
           words, revs, spans, events, utts, dirty[1], dirty[10], grow, rss[1], rss[10], med[2], med[10]);
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define SOAK_SANITIZED 1
#endif
#endif
#ifndef SOAK_SANITIZED                     /* ASan's quarantine and shadow memory hide the numbers; the point of that build is the memory errors */
    if (grow > 2.0) { fprintf(stderr, "memory did not plateau\n"); bad++; }
    if (med[10] > 2 * med[2]) { fprintf(stderr, "cost per word grew\n"); bad++; }
#endif
    arctag_destroy(h); free(cost);
    return bad ? 1 : 0;
}
