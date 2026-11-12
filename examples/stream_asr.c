/* stream_asr: drive arctag's streaming API from an ASR word stream (push / poll), the way a live captioning or agent-assist
 * app does it. Input: JSON Lines of ASR messages (examples/data/asr_stream.jsonl):
 *   {"type": "interim", "words": [{"w": "set", "t0": 0.10, "t1": 0.31, "conf": 0.82, "spk": "A"}, ...]}  unstable hypothesis
 *   {"type": "final",   "words": [...]}      the stable words of the current segment (replace the interim ones)
 *   {"type": "endpoint"}                      end of utterance (VAD / speech_final / end of turn)
 * Interim words are revisions: each interim / final message replaces everything after the last final word (push_words
 * with "from"). After each message the app polls and keeps a span list: spans with end > a0 are replaced by the poll's.
 *   stream_asr [-m MODEL_DIR] [-c CONFIG.json] [--int8] [-d DOMAIN] [--every N] [INPUT.jsonl|-]
 * --every N polls after every N messages (default 1). Prints one line per poll: n_words, new events, closed utterances
 * (intent / act / safety that are on), and at the end the final span list. */
#include "arctag.h"
#include "cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_line(FILE *f) {
    size_t cap = 4096, n = 0; char *b = (char *)malloc(cap); int c;
    while ((c = fgetc(f)) != EOF && c != '\n') { if (n + 1 >= cap) { cap *= 2; b = (char *)realloc(b, cap); } b[n++] = (char)c; }
    if (c == EOF && n == 0) { free(b); return NULL; }
    b[n] = 0; return b;
}

static void print_on(const char *label, cJSON *u, const char *key) {
    cJSON *a = cJSON_GetObjectItem(u, key);
    for (cJSON *x = a ? a->child : NULL; x; x = x->next)
        if (cJSON_IsTrue(cJSON_GetObjectItem(x, "on"))) printf(" %s=%s", label, cJSON_GetObjectItem(x, "tag")->valuestring);
}

int main(int argc, char **argv) {
    const char *model = NULL, *config = NULL, *domain = NULL, *in = "-";
    int int8 = 0, every = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) config = argv[++i];
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) domain = argv[++i];
        else if (!strcmp(argv[i], "--every") && i + 1 < argc) every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--int8")) int8 = 1;
        else if (argv[i][0] == '-' && argv[i][1]) { fprintf(stderr, "usage: %s [-m MODEL_DIR] [-c CONFIG.json] [--int8] [-d DOMAIN] [--every N] [INPUT|-]\n", argv[0]); return 2; }
        else in = argv[i];
    }
    if (every < 1) every = 1;
    char err[512];
    arctag_t *h;
    if (config) h = arctag_create_from_config(model, config, err, sizeof err);
    else { arctag_options o; arctag_options_init(&o); o.precision = int8 ? ARCTAG_INT8 : ARCTAG_FP32; h = arctag_create_ex(model ? model : "model", &o, err, sizeof err); }
    if (!h) { fprintf(stderr, "arctag: %s\n", err); return 1; }
    if ((config && int8 && arctag_set_option(h, "precision", "int8")) || (domain && arctag_set_option(h, "domain", domain))) {
        fprintf(stderr, "option: %s\n", arctag_last_error(h)); arctag_destroy(h); return 1;
    }
    FILE *f = strcmp(in, "-") ? fopen(in, "rb") : stdin;
    if (!f) { fprintf(stderr, "cannot open %s\n", in); arctag_destroy(h); return 1; }
    cJSON *spans = cJSON_CreateArray();         /* the app's span list */
    int n_final = 0, msgs = 0; char *line;
    while ((line = read_line(f)) != NULL) {
        cJSON *m = cJSON_Parse(line); free(line);
        if (!m) continue;
        const char *type = cJSON_GetStringValue(cJSON_GetObjectItem(m, "type"));
        cJSON *req = cJSON_CreateObject();
        if (type && !strcmp(type, "endpoint")) cJSON_AddBoolToObject(req, "endpoint", 1);
        else {
            cJSON *w = cJSON_GetObjectItem(m, "words"); int final = type && !strcmp(type, "final");
            cJSON *ws = cJSON_AddArrayToObject(req, "words");
            for (cJSON *x = w ? w->child : NULL; x; x = x->next) {
                cJSON *c = cJSON_Duplicate(x, 1);
                if (cJSON_IsObject(c)) cJSON_AddBoolToObject(c, "final", final);
                cJSON_AddItemToArray(ws, c);
            }
            cJSON_AddNumberToObject(req, "from", n_final);           /* revise everything after the stable words */
            if (final) n_final += cJSON_GetArraySize(w);
        }
        char *js = cJSON_PrintUnformatted(req); cJSON_Delete(req); cJSON_Delete(m);
        if (arctag_push_words(h, js)) fprintf(stderr, "push: %s\n", arctag_last_error(h));
        free(js);
        if (++msgs % every) continue;
        char *out = arctag_poll(h);
        cJSON *p = out ? cJSON_Parse(out) : NULL; arctag_free_str(out);
        if (!p) { fprintf(stderr, "poll: %s\n", arctag_last_error(h)); break; }
        int a0 = cJSON_GetObjectItem(p, "a0")->valueint;
        for (cJSON *x = spans->child; x;) {                           /* drop our spans that end after a0 ... */
            cJSON *nx = x->next;
            if (cJSON_GetObjectItem(x, "end")->valueint > a0) cJSON_Delete(cJSON_DetachItemViaPointer(spans, x));
            x = nx;
        }
        cJSON *ps = cJSON_GetObjectItem(p, "spans"), *x;             /* ... and take the poll's */
        while ((x = cJSON_DetachItemFromArray(ps, 0)) != NULL) cJSON_AddItemToArray(spans, x);
        printf("poll n_words=%d profile=%s", cJSON_GetObjectItem(p, "n_words")->valueint, cJSON_GetStringValue(cJSON_GetObjectItem(p, "profile")));
        for (cJSON *e = cJSON_GetObjectItem(p, "events")->child; e; e = e->next)
            printf(" event=%s@%d", cJSON_GetObjectItem(e, "type")->valuestring, cJSON_GetObjectItem(e, "word")->valueint);
        for (cJSON *u = cJSON_GetObjectItem(p, "utterances")->child; u; u = u->next) {
            printf("\n  utterance [%d,%d)", cJSON_GetObjectItem(u, "start")->valueint, cJSON_GetObjectItem(u, "end")->valueint);
            print_on("intent", u, "intent"); print_on("act", u, "act"); print_on("safety", u, "safety");
        }
        cJSON *rt = cJSON_GetObjectItem(p, "redacted_text");
        if (cJSON_IsString(rt)) printf("\n  caption: %s", rt->valuestring);
        printf("\n");
        cJSON_Delete(p);
    }
    if (f != stdin) fclose(f);
    printf("final spans:\n");
    for (cJSON *s = spans->child; s; s = s->next)
        printf("  [%d,%d) %-32s %s\n", cJSON_GetObjectItem(s, "start")->valueint, cJSON_GetObjectItem(s, "end")->valueint,
               cJSON_GetObjectItem(s, "type")->valuestring, cJSON_GetObjectItem(s, "text")->valuestring);
    cJSON_Delete(spans);
    arctag_destroy(h);
    return 0;
}
