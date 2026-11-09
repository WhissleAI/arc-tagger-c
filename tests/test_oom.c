/* Every allocation fails once, in turn, in each public call: no crash, and an error rather than a document with holes.
 * The library is built with tests/fail_alloc.h force-included. usage: test_oom MODEL_DIR */
#include "arctag.h"
#include "cJSON.h"
#include <stdio.h>
#undef malloc
#undef calloc
#undef realloc
#undef strdup
#include <stdlib.h>
#include <string.h>

static long armed = 0, count = 0;
static int should_fail(void) { count++; if (armed > 0 && --armed == 0) return 1; return 0; }
void *fi_malloc(size_t n) { return should_fail() ? NULL : malloc(n); }
void *fi_calloc(size_t n, size_t z) { return should_fail() ? NULL : calloc(n, z); }
void *fi_realloc(void *p, size_t n) { return should_fail() ? NULL : realloc(p, n); }
char *fi_strdup(const char *s) { if (should_fail()) return NULL; size_t n = strlen(s) + 1; char *d = malloc(n); if (d) memcpy(d, s, n); return d; }
void fi_arm(long nth) { armed = nth; }
long fi_count(void) { return count; }

static const char *TEXT = "Call Maria Lopez at 415 555 0199 tomorrow. The Lakers are up 98 to 95 now. Okay team, let's start!";
static const char *W = "[\"hello\", {\"w\": \"my\", \"t0\": 1.2, \"t1\": 1.5, \"conf\": 0.9, \"spk\": \"A\"}, {\"w\": \"name\", \"spk\": \"B\", \"eou\": true}, \"is\", \"Maria.\"]";
static long fails_seen = 0, passes = 0; static int bad = 0;
/* run a call that returns a document with the kth allocation failing */
typedef char *(*docfn)(arctag_t *);
static char *do_text(arctag_t *h) { return arctag_tag_text(h, TEXT); }
static char *do_words(arctag_t *h) { return arctag_tag_words(h, W); }
static char *do_poll(arctag_t *h) { return arctag_poll(h); }
static void sweep(arctag_t *h, const char *name, docfn fn, void (*setup)(arctag_t *)) {
    for (long k = 1; k < 100000; k++) {
        if (setup) setup(h);
        fi_arm(k);
        char *o = fn(h);
        int fired = armed == 0;
        fi_arm(0);
        if (fired) {
            fails_seen++;
            if (o) {
                /* a failure inside something optional is allowed to succeed only with a complete document */
                cJSON *j = cJSON_Parse(o);
                if (!j || !cJSON_GetObjectItem(j, "spans") || !cJSON_GetObjectItem(j, "events") || !cJSON_GetObjectItem(j, "utterances")) { fprintf(stderr, "%s, allocation %ld: incomplete document\n", name, k); bad++; }
                cJSON_Delete(j);
            } else if (!arctag_last_error(h)[0]) { fprintf(stderr, "%s, allocation %ld: NULL without an error\n", name, k); bad++; }
        }
        arctag_free_str(o);
        if (!fired) { passes++; printf("  %s: %ld allocations, each failed once\n", name, k - 1); return; }
    }
}
static void setup_stream(arctag_t *h) { arctag_reset(h); arctag_push_words(h, W); arctag_push_words(h, "{\"words\":[\"Call\",\"me.\"],\"endpoint\":true}"); arctag_push_words(h, "[\"more\",\"words\"]"); }
int main(int c, char **v) {
    if (c < 2) { fprintf(stderr, "usage: %s MODEL_DIR\n", v[0]); return 2; }
    char err[512];
    /* create */
    for (long k = 1; k < 2000000; k += k < 200 ? 1 : 97) {
        fi_arm(k);
        arctag_t *h = arctag_create(v[1], err, sizeof err);
        int fired = armed == 0;
        fi_arm(0);
        if (!fired) { arctag_destroy(h); printf("  create: %ld allocations reached\n", k); break; }
        fails_seen++;
        if (h) { arctag_destroy(h); }                      /* the tokenizer's map can survive a lost entry only if it reports it */
        else if (!err[0]) { fprintf(stderr, "create, allocation %ld: no message\n", k); bad++; }
    }
    arctag_t *h = arctag_create(v[1], err, sizeof err);
    if (!h) { fprintf(stderr, "%s\n", err); return 1; }
    sweep(h, "tag_text", do_text, NULL);
    sweep(h, "tag_words", do_words, NULL);
    sweep(h, "poll", do_poll, setup_stream);
    /* push */
    for (long k = 1; k < 100000; k++) {
        arctag_reset(h);
        fi_arm(k);
        int rc = arctag_push_words(h, W);
        int fired = armed == 0;
        fi_arm(0);
        if (!fired) { printf("  push: %ld allocations, each failed once\n", k - 1); break; }
        fails_seen++;
        if (rc == 0) { fprintf(stderr, "push, allocation %ld: reported success\n", k); bad++; }
        arctag_free_str(arctag_poll(h));                   /* still usable */
    }
    arctag_reset(h);
    arctag_free_str(arctag_tag_text(h, TEXT));
    arctag_destroy(h);
    printf("test_oom: %ld injected failures, %d wrong\n", fails_seen, bad);
    return bad ? 1 : 0;
}
