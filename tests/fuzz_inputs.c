/* Text and JSON inputs: no crash, valid JSON out, and the API keeps working. Run under ASan + UBSan by scripts/test_native.sh.
 * A deterministic mutation fuzzer, because Apple's clang ships no libFuzzer. usage: fuzz_inputs MODEL_DIR ITERATIONS [SEED] */
#include "arctag.h"
#include "cJSON.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t s = 0x2545F4914F6CDD1Dull;
static uint64_t rnd(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
static const char *SEEDS[] = {
    "[\"hello\", \"world.\"]",
    "[\"a\", {\"w\": \"word\", \"t0\": 1.2, \"t1\": 1.5, \"conf\": 0.9, \"spk\": \"A\", \"final\": true, \"eou\": true}, {\"word\": \"x\", \"start\": 2, \"end\": 3, \"confidence\": 0.1, \"speaker\": 3, \"final\": false}]",
    "{\"words\": [\"John\", \"Smith.\"], \"endpoint\": true}",
    "{\"words\": [\"replaced\"], \"from\": 1}",
    "{\"words\": [], \"from\": 0}",
    "{\"endpoint\": true}",
    "{\"from\": -5, \"words\": [\"x\"]}",
    "{\"from\": 2147483648, \"words\": [\"x\"]}",
    "{\"from\": 1e300, \"words\": [\"x\"]}",
    "[{\"w\": \"t\", \"t0\": 1e308, \"t1\": -1e308, \"conf\": 1e308}, {\"w\": \"u\", \"t0\": -1e308, \"t1\": 1e308, \"conf\": -1e308}, {\"w\": \"v\", \"t0\": 0, \"t1\": 1e300}]",
    "[{\"w\": \"s\", \"spk\": \"\"}, {\"w\": \"s\", \"spk\": 2147483647}, {\"w\": \"s\", \"speaker\": -1}, {\"w\": \"\", \"spk\": \"A\"}, {\"w\": \"   \"}, {\"text\": \"\\u0000x\"}]",
    "[\"\\ud83d\\ude00 emoji\", \"caf\\u00e9\", \"\\u0663\\u0664\", \"415-555-0199\", \"<|endoftext|>\", \"\\ud800\", \"tab\\tbed\"]",
    "[1, null, true, [], {}, {\"w\": 5}, \"ok\"]",
    "{\"words\": \"not an array\"}",
    "{\"words\": {\"w\": \"x\"}}",
    "[[[[[[[[[[[[[[[[\"deep\"]]]]]]]]]]]]]]]]",
    "not json at all",
    "",
};
static const char *TEXTS[] = {
    "Hi, this is Maria Lopez. My number is 555 867 5309. The Lakers are up 98 to 95!",
    "", "   ", "\t\n\r\f\v", ".", "...", "? ! .", "\"quoted.\" (paren.) [x.] {y.}",
    "\xff\xfe\xfd invalid \xc0\x80 utf8 \xed\xa0\x80 \xf4\x90\x80\x80 \xe2\x82",
    "<|endoftext|> special <|endoftext|><|endoftext|>",
};
#define NSEEDS (sizeof SEEDS / sizeof *SEEDS)
#define NTEXTS (sizeof TEXTS / sizeof *TEXTS)
static const char *INSERTS[] = {"null", "1e999", "-1", "\"", "{", "}", "[", "]", ",", ":", "\\", "\\u0000", "\xff", "\"from\":0,", "\"spk\":\"Z\",", "true", "\"eou\":true,", "0.0000001"};
static char *mutate(const char *seed, size_t extra) {
    size_t n = strlen(seed), cap = n + extra + 64; char *b = malloc(cap); memcpy(b, seed, n + 1);
    int edits = (int)(rnd() % 5);
    for (int e = 0; e < edits; e++) {
        n = strlen(b);
        switch (rnd() % 4) {
            case 0: if (n) { size_t at = rnd() % n; b[at] = (char)(1 + rnd() % 255); } break;
            case 1: if (n) { size_t at = rnd() % n; memmove(b + at, b + at + 1, n - at); } break;
            case 2: { const char *ins = INSERTS[rnd() % (sizeof INSERTS / sizeof *INSERTS)]; size_t L = strlen(ins), at = n ? rnd() % (n + 1) : 0; if (n + L + 1 < cap) { memmove(b + at + L, b + at, n - at + 1); memcpy(b + at, ins, L); } break; }
            case 3: if (n > 2) { size_t cut = 1 + rnd() % (n - 1); b[cut] = 0; } break;
        }
    }
    return b;
}
static char *random_words(int nmax) {
    int n = (int)(rnd() % (uint64_t)nmax); size_t cap = (size_t)n * 40 + 16, k = 0; char *b = malloc(cap);
    for (int i = 0; i < n; i++) {
        int L = 1 + (int)(rnd() % 12);
        for (int j = 0; j < L; j++) { unsigned char ch = (unsigned char)(rnd() & 1 ? 'a' + rnd() % 26 : 1 + rnd() % 255); if (ch == 0) ch = 'x'; b[k++] = (char)ch; }
        b[k++] = rnd() % 8 ? ' ' : '.'; if (b[k - 1] == '.') b[k++] = ' ';
    }
    b[k] = 0;
    return b;
}
static long ok_docs = 0, null_docs = 0, ok_push = 0, bad_push = 0; static int bad = 0;
static void check_doc(arctag_t *h, char *o, const char *what) {
    if (!o) { null_docs++; if (!arctag_last_error(h)[0]) { fprintf(stderr, "%s returned NULL with no error\n", what); bad++; } return; }
    cJSON *j = cJSON_Parse(o);
    if (!j) { fprintf(stderr, "%s returned invalid JSON: %.200s\n", what, o); bad++; }
    else {
        int poll = !strcmp(what, "poll");
        const char *need[] = {"spans", "events", "utterances"};
        for (int i = 0; i < 3; i++) if (!cJSON_IsArray(cJSON_GetObjectItem(j, need[i]))) { fprintf(stderr, "%s: no array %s\n", what, need[i]); bad++; }
        if (poll && !cJSON_GetObjectItem(j, "current")) { fprintf(stderr, "poll: no current\n"); bad++; }
        ok_docs++;
    }
    cJSON_Delete(j); arctag_free_str(o);
}
int main(int c, char **v) {
    if (c < 3) { fprintf(stderr, "usage: %s MODEL_DIR ITERATIONS [SEED]\n", v[0]); return 2; }
    long iters = atol(v[2]); if (c > 3) s ^= strtoull(v[3], NULL, 10) * 0x9E3779B97F4A7C15ull; if (!s) s = 1;
    char err[512]; arctag_t *h = arctag_create(v[1], err, sizeof err); if (!h) { fprintf(stderr, "%s\n", err); return 1; }
    /* NULLs and odd arguments */
    arctag_free_str(arctag_tag_text(NULL, "x")); arctag_free_str(arctag_tag_text(h, NULL)); arctag_free_str(arctag_tag_words(h, NULL)); arctag_free_str(arctag_poll(NULL));
    if (arctag_push_words(NULL, "[]") != -1 || arctag_push_words(h, NULL) != -1 || arctag_set_option(h, NULL, "x") != -1 || arctag_set_option(h, "x", NULL) != -1) { fprintf(stderr, "NULL arguments\n"); bad++; }
    arctag_reset(NULL); arctag_destroy(NULL); arctag_free_str(NULL);
    if (arctag_create(NULL, err, sizeof err) || arctag_create("/nonexistent", NULL, 0)) { fprintf(stderr, "bad create\n"); bad++; }
    /* very long inputs */
    { size_t L = 300000; char *big = malloc(L + 8); memset(big, 'a', L); big[L] = 0; check_doc(h, arctag_tag_text(h, big), "tag_text");
      char *js = malloc(L + 16); memset(js, 'b', L); js[0] = '['; js[1] = '"'; js[L - 2] = '"'; js[L - 1] = ']'; js[L] = 0;
      arctag_push_words(h, js); check_doc(h, arctag_poll(h), "poll"); arctag_reset(h); free(js); free(big);
      /* more words than the backlog limit: refused, stream intact */
      size_t N = 70000; char *arr = malloc(N * 4 + 8); size_t k = 0; arr[k++] = '['; for (size_t i = 0; i < N; i++) { arr[k++] = '"'; arr[k++] = 'x'; arr[k++] = '"'; arr[k++] = ','; } arr[k - 1] = ']'; arr[k] = 0;
      if (arctag_push_words(h, arr) != -1) { fprintf(stderr, "backlog limit not enforced\n"); bad++; } free(arr);
      /* many words in one text: split into chunks; a large backlog through one poll */
      size_t n3 = 2100; char *t3 = malloc(n3 * 2 + 1); for (size_t i = 0; i < n3; i++) { t3[2 * i] = 'w'; t3[2 * i + 1] = ' '; } t3[n3 * 2] = 0; check_doc(h, arctag_tag_text(h, t3), "tag_text"); free(t3); }
    const char *doms[] = {"auto", "meeting", "sports", "no_such_domain", "", "nba", "ui:fine"};
    const char *keys[] = {"domain", "asr", "threads", "max_ctx", "stream_ctx", "unknown"};
    const char *vals[] = {"16", "2048", "15", "2049", "-1", "abc", "", "gold", "unknown", "web", "300"};
    for (long i = 0; i < iters; i++) {
        switch (rnd() % 8) {
            case 0: { char *m = mutate(TEXTS[rnd() % NTEXTS], 256); check_doc(h, arctag_tag_text(h, m), "tag_text"); free(m); break; }
            case 1: { char *m = random_words(40); check_doc(h, arctag_tag_text(h, m), "tag_text"); free(m); break; }
            case 2: { char *m = mutate(SEEDS[rnd() % NSEEDS], 256); check_doc(h, arctag_tag_words(h, m), "tag_words"); free(m); break; }
            case 3: case 4: case 5: { char *m = mutate(SEEDS[rnd() % NSEEDS], 256); if (arctag_push_words(h, m)) bad_push++; else ok_push++; free(m); if (rnd() % 2) check_doc(h, arctag_poll(h), "poll"); break; }
            case 6: check_doc(h, arctag_poll(h), "poll"); break;
            case 7:
                switch (rnd() % 4) {
                    case 0: arctag_reset(h); break;
                    case 1: arctag_set_option(h, "domain", doms[rnd() % (sizeof doms / sizeof *doms)]); break;
                    case 2: arctag_set_option(h, keys[rnd() % 6], vals[rnd() % 11]); break;
                    case 3: { char js[64]; snprintf(js, sizeof js, "{\"from\":%d,\"words\":[\"r\",\"s\"]}", (int)(rnd() % 600) - 50); arctag_push_words(h, js); break; }
                }
                break;
        }
    }
    check_doc(h, arctag_poll(h), "poll");
    /* A long session with valid input and random behaviour, which the mutated JSON above rarely keeps alive: the held-word window moves,
     * revisions land at every depth including deep replays and too-old ones, the window size changes, polls come at random cadence. */
    arctag_reset(h); arctag_set_option(h, "domain", "meeting");
    long n = 0, too_old = 0, deep = 0;
    for (long i = 0; i < iters; i++) {
        char js[8192]; size_t k = 0; int cnt = 1 + (int)(rnd() % (rnd() % 16 ? 6 : 200));
        if (cnt > 60) cnt = 60;
        int rev = rnd() % 4 == 0; long from = n;
        if (rev) { long depth = (rnd() % 64 || n < 1000) ? (long)(rnd() % (rnd() % 8 ? 6 : 60)) : (long)(rnd() % 1500); from = n - depth; if (from < 0) from = 0; if (depth > 64) deep++; }
        k += (size_t)snprintf(js + k, sizeof js - k, "{\"from\":%ld,\"endpoint\":%s,\"words\":[", from, rnd() % 16 ? "false" : "true");
        for (int w = 0; w < cnt; w++) {
            unsigned r = (unsigned)(rnd() >> 32);
            k += (size_t)snprintf(js + k, sizeof js - k, "%s{\"w\":\"w%u%s\",\"t0\":%.2f,\"t1\":%.2f,\"spk\":\"S%u\",\"eou\":%s}", w ? "," : "", r % 500, r % 11 ? "" : ".", (double)(from + w) * 0.3, (double)(from + w) * 0.3 + 0.2, r % 19, r % 13 ? "false" : "true");
        }
        k += (size_t)snprintf(js + k, sizeof js - k, "]}");
        if (arctag_push_words(h, js) == 0) n = from + cnt;
        else { too_old++; if (!strstr(arctag_last_error(h), "no longer held")) { fprintf(stderr, "long session: unexpected refusal: %s\n", arctag_last_error(h)); bad++; } }
        if (rnd() % 3 == 0) {
            char *o = arctag_poll(h); cJSON *j = o ? cJSON_Parse(o) : NULL;
            cJSON *nw = j ? cJSON_GetObjectItem(j, "n_words") : NULL;
            if (!nw || nw->valueint != (int)n) { fprintf(stderr, "long session: n_words wrong at %ld\n", n); bad++; }
            cJSON_Delete(j); arctag_free_str(o);
        }
        if (rnd() % 400 == 0) { char v2[16]; snprintf(v2, sizeof v2, "%d", 16 + (int)(rnd() % 400)); arctag_set_option(h, "stream_ctx", v2); }
    }
    check_doc(h, arctag_poll(h), "poll");
    printf("fuzz_inputs: long session: %ld words, %ld deep revisions, %ld refused as too old\n", n, deep, too_old);
    arctag_destroy(h);
    printf("fuzz_inputs: %ld iterations, %ld documents valid, %ld NULL with an error, %ld pushes accepted, %ld refused, %d wrong\n", iters, ok_docs, null_docs, ok_push, bad_push, bad);
    return bad ? 1 : 0;
}
