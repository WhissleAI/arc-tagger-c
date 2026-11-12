/* vertical_run: run one vertical config (configs/NAME.json) over an input and print a compact, vertical-oriented report.
 *   vertical_run CONFIG.json [-m MODEL_DIR] [--int8] [--backend B] [--json] [INPUT|-]
 * INPUT: plain text, or JSON Lines of words / ASR words ({"w","t0","t1","spk",...} per line, {"endpoint": true} lines).
 * The config's "mode" decides how the input is fed: "batch" = arctag_tag_text / arctag_tag_words over the whole input;
 * "streaming" = utterance by utterance through arctag_push_words + arctag_poll (as a live app would).
 * The report lists per utterance the "on" read-outs, then spans by family, events, and the redacted text when the config
 * redacts. --json prints the raw library JSON instead (the last poll / the batch result). */
#include "arctag.h"
#include "cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *slurp(FILE *f) {
    size_t cap = 1 << 16, n = 0; char *b = (char *)malloc(cap); size_t k;
    while ((k = fread(b + n, 1, cap - n - 1, f)) > 0) { n += k; if (n + 1 >= cap) { cap *= 2; b = (char *)realloc(b, cap); } }
    b[n] = 0; return b;
}

static void on_list(cJSON *u, const char *key) {
    cJSON *a = cJSON_GetObjectItem(u, key); int first = 1;
    for (cJSON *x = a ? a->child : NULL; x; x = x->next)
        if (cJSON_IsTrue(cJSON_GetObjectItem(x, "on"))) { printf("%s%s", first ? "" : ",", cJSON_GetObjectItem(x, "tag")->valuestring); first = 0; }
    if (first && a) printf("-");
}

static void report_utt(cJSON *u) {
    static const char *keys[] = {"intent", "act", "safety", "role", "emotion", "valence", "item"};
    printf("  [%3d,%3d)", cJSON_GetObjectItem(u, "start")->valueint, cJSON_GetObjectItem(u, "end")->valueint);
    for (int k = 0; k < 7; k++) if (cJSON_GetObjectItem(u, keys[k])) { printf(" %s=", keys[k]); on_list(u, keys[k]); }
    cJSON *st = cJSON_GetObjectItem(u, "state");
    for (cJSON *g = st ? st->child : NULL; g; g = g->next) {
        cJSON *t = g->child;
        if (t && cJSON_IsTrue(cJSON_GetObjectItem(t, "on"))) printf(" %s=%s", g->string, cJSON_GetObjectItem(t, "tag")->valuestring);
    }
    printf("\n");
}

static void report(cJSON *spans, cJSON *events, cJSON *utts, const char *redacted) {
    printf("utterances:\n");
    for (cJSON *u = utts ? utts->child : NULL; u; u = u->next) report_utt(u);
    printf("spans:\n");
    for (cJSON *s = spans ? spans->child : NULL; s; s = s->next)
        printf("  %-9s %-34s [%d,%d) %s\n", cJSON_GetObjectItem(s, "family")->valuestring, cJSON_GetObjectItem(s, "type")->valuestring,
               cJSON_GetObjectItem(s, "start")->valueint, cJSON_GetObjectItem(s, "end")->valueint, cJSON_GetObjectItem(s, "text")->valuestring);
    printf("events:\n");
    for (cJSON *e = events ? events->child : NULL; e; e = e->next)
        printf("  %-40s word %d (%s)\n", cJSON_GetObjectItem(e, "type")->valuestring, cJSON_GetObjectItem(e, "word")->valueint,
               cJSON_GetObjectItem(e, "text")->valuestring);
    if (redacted) printf("redacted:\n  %s\n", redacted);
}

int main(int argc, char **argv) {
    const char *model = NULL, *config = NULL, *in = "-";
    const char *backend = NULL;
    int int8 = 0, json = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--int8")) int8 = 1;
        else if (!strcmp(argv[i], "--backend") && i + 1 < argc) backend = argv[++i];
        else if (!strcmp(argv[i], "--json")) json = 1;
        else if (argv[i][0] == '-' && argv[i][1]) { fprintf(stderr, "usage: %s CONFIG.json [-m MODEL_DIR] [--int8] [--backend B] [--json] [INPUT|-]\n", argv[0]); return 2; }
        else if (!config) config = argv[i];
        else in = argv[i];
    }
    if (!config) { fprintf(stderr, "usage: %s CONFIG.json [-m MODEL_DIR] [--int8] [--backend B] [--json] [INPUT|-]\n", argv[0]); return 2; }
    FILE *cf = fopen(config, "rb"); if (!cf) { fprintf(stderr, "cannot open %s\n", config); return 1; }
    char *ctext = slurp(cf); fclose(cf);
    cJSON *cfg = cJSON_Parse(ctext); free(ctext);
    const char *mode = cJSON_GetStringValue(cJSON_GetObjectItem(cfg, "mode"));
    int streaming = mode && !strcmp(mode, "streaming");
    char err[512];
    arctag_t *h = arctag_create_from_config(model, config, err, sizeof err);
    if (!h) { fprintf(stderr, "arctag: %s\n", err); cJSON_Delete(cfg); return 1; }
    if (backend && arctag_set_option(h, "backend", backend)) { fprintf(stderr, "backend: %s\n", arctag_last_error(h)); arctag_destroy(h); return 1; }
    if (int8 && arctag_set_option(h, "precision", "int8")) { fprintf(stderr, "int8: %s\n", arctag_last_error(h)); return 1; }
    FILE *f = strcmp(in, "-") ? fopen(in, "rb") : stdin;
    if (!f) { fprintf(stderr, "cannot open %s\n", in); return 1; }
    char *input = slurp(f); if (f != stdin) fclose(f);
    char *info = arctag_info(h);
    if (!json) printf("config %s (%s): %s\n", cJSON_GetStringValue(cJSON_GetObjectItem(cfg, "name")), streaming ? "streaming" : "batch", info);
    arctag_free_str(info);
    /* input: JSON Lines of words, or plain text */
    int is_words = input[strspn(input, " \t\r\n")] == '{';
    cJSON *words = NULL;
    if (is_words) {
        words = cJSON_CreateArray();
        for (char *l = strtok(input, "\n"); l; l = strtok(NULL, "\n")) {
            cJSON *w = cJSON_Parse(l); if (!w) continue;
            if (cJSON_IsTrue(cJSON_GetObjectItem(w, "endpoint"))) {     /* end of utterance: mark the previous word */
                cJSON *last = cJSON_GetArrayItem(words, cJSON_GetArraySize(words) - 1);
                if (last) cJSON_AddBoolToObject(last, "eou", 1);
                cJSON_Delete(w);
            } else cJSON_AddItemToArray(words, w);
        }
    }
    if (!streaming) {
        char *res;
        if (words) { char *ws = cJSON_PrintUnformatted(words); res = arctag_tag_words(h, ws); free(ws); }
        else res = arctag_tag_text(h, input);
        if (!res) { fprintf(stderr, "error: %s\n", arctag_last_error(h)); return 1; }
        if (json) puts(res);
        else {
            cJSON *r = cJSON_Parse(res);
            printf("profile %s, %d words\n", cJSON_GetStringValue(cJSON_GetObjectItem(r, "profile")), cJSON_GetObjectItem(r, "n_words")->valueint);
            report(cJSON_GetObjectItem(r, "spans"), cJSON_GetObjectItem(r, "events"), cJSON_GetObjectItem(r, "utterances"),
                   cJSON_GetStringValue(cJSON_GetObjectItem(r, "redacted_text")));
            cJSON_Delete(r);
        }
        arctag_free_str(res);
    } else {
        /* streaming: push one utterance at a time (text: split after . ? !), poll after each */
        cJSON *spans = cJSON_CreateArray(), *events = cJSON_CreateArray(), *utts = cJSON_CreateArray(), *last = NULL;
        cJSON *cur = cJSON_CreateArray();
        cJSON *src = words;
        if (!src) {
            src = cJSON_CreateArray();
            for (char *t = strtok(input, " \t\r\n"); t; t = strtok(NULL, " \t\r\n")) {
                cJSON *w = cJSON_CreateObject(); cJSON_AddStringToObject(w, "w", t);
                size_t L = strlen(t); while (L && strchr("\"')]}", t[L - 1])) L--;
                if (L && strchr(".?!", t[L - 1])) cJSON_AddBoolToObject(w, "eou", 1);
                cJSON_AddItemToArray(src, w);
            }
        }
        int n = cJSON_GetArraySize(src), i = 0;
        for (cJSON *w = src->child; w; w = w->next, i++) {
            cJSON_AddItemToArray(cur, cJSON_Duplicate(w, 1));
            if (!cJSON_IsTrue(cJSON_GetObjectItem(w, "eou")) && i != n - 1) continue;
            char *ws = cJSON_PrintUnformatted(cur); cJSON_Delete(cur); cur = cJSON_CreateArray();
            if (arctag_push_words(h, ws)) fprintf(stderr, "push: %s\n", arctag_last_error(h));
            free(ws);
            if (i == n - 1) arctag_push_words(h, "{\"endpoint\": true}");
            char *out = arctag_poll(h);
            cJSON *p = out ? cJSON_Parse(out) : NULL; arctag_free_str(out);
            if (!p) break;
            int a0 = cJSON_GetObjectItem(p, "a0")->valueint;
            for (cJSON *x = spans->child; x;) { cJSON *nx = x->next; if (cJSON_GetObjectItem(x, "end")->valueint > a0) cJSON_Delete(cJSON_DetachItemViaPointer(spans, x)); x = nx; }
            cJSON *x;
            while ((x = cJSON_DetachItemFromArray(cJSON_GetObjectItem(p, "spans"), 0))) cJSON_AddItemToArray(spans, x);
            while ((x = cJSON_DetachItemFromArray(cJSON_GetObjectItem(p, "events"), 0))) cJSON_AddItemToArray(events, x);
            while ((x = cJSON_DetachItemFromArray(cJSON_GetObjectItem(p, "utterances"), 0))) cJSON_AddItemToArray(utts, x);
            cJSON_Delete(last); last = p;
        }
        if (json) { char *s = cJSON_PrintUnformatted(last); puts(s ? s : "null"); free(s); }
        else {
            printf("profile %s, %d words (streamed)\n", last ? cJSON_GetStringValue(cJSON_GetObjectItem(last, "profile")) : "?", n);
            report(spans, events, utts, last ? cJSON_GetStringValue(cJSON_GetObjectItem(last, "redacted_text")) : NULL);
        }
        cJSON_Delete(last); cJSON_Delete(spans); cJSON_Delete(events); cJSON_Delete(utts); cJSON_Delete(cur);
        if (src != words) cJSON_Delete(src);
    }
    cJSON_Delete(words); cJSON_Delete(cfg); free(input);
    arctag_destroy(h);
    return 0;
}
