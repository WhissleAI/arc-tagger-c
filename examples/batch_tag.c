/* batch_tag: tag many documents, one JSON result per input line (JSON Lines out).
 *   batch_tag [-m MODEL_DIR] [-c CONFIG.json] [--int8] [-d DOMAIN] [-t THREADS] [INPUT.txt|-] > out.jsonl
 * Input: one document per line, plain text, or a JSON object {"id": ..., "text": "..."} / {"id": ..., "words": [...]}.
 * Output: {"id": <id or line number>, "ms": <tagging time>, "result": <arctag_tag_text / arctag_tag_words JSON>} per line;
 * a summary (documents, words, ms per word) goes to stderr. */
#include "arctag.h"
#include "cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ex_time.h"

static char *read_line(FILE *f) {
    size_t cap = 4096, n = 0; char *b = (char *)malloc(cap); int c;
    while ((c = fgetc(f)) != EOF && c != '\n') { if (n + 1 >= cap) { cap *= 2; b = (char *)realloc(b, cap); } b[n++] = (char)c; }
    if (c == EOF && n == 0) { free(b); return NULL; }
    if (n && b[n - 1] == '\r') n--;
    b[n] = 0; return b;
}

int main(int argc, char **argv) {
    const char *model = NULL, *config = NULL, *domain = NULL, *threads = NULL, *in = "-";
    int int8 = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) config = argv[++i];
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) domain = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = argv[++i];
        else if (!strcmp(argv[i], "--int8")) int8 = 1;
        else if (argv[i][0] == '-' && argv[i][1]) { fprintf(stderr, "usage: %s [-m MODEL_DIR] [-c CONFIG.json] [--int8] [-d DOMAIN] [-t THREADS] [INPUT|-]\n", argv[0]); return 2; }
        else in = argv[i];
    }
    char err[512];
    arctag_t *h;
    if (config) h = arctag_create_from_config(model, config, err, sizeof err);
    else { arctag_options o; arctag_options_init(&o); o.precision = int8 ? ARCTAG_INT8 : ARCTAG_FP32; h = arctag_create_ex(model ? model : "model", &o, err, sizeof err); }
    if (!h) { fprintf(stderr, "arctag: %s\n", err); return 1; }
    if ((config && int8 && arctag_set_option(h, "precision", "int8")) || (domain && arctag_set_option(h, "domain", domain)) ||
        (threads && arctag_set_option(h, "threads", threads))) { fprintf(stderr, "option: %s\n", arctag_last_error(h)); arctag_destroy(h); return 1; }
    FILE *f = strcmp(in, "-") ? fopen(in, "rb") : stdin;
    if (!f) { fprintf(stderr, "cannot open %s\n", in); arctag_destroy(h); return 1; }
    long docs = 0, words = 0; double total = 0; char *line;
    while ((line = read_line(f)) != NULL) {
        if (!*line) { free(line); continue; }
        cJSON *j = line[0] == '{' ? cJSON_Parse(line) : NULL, *id = NULL;
        double t0 = now_ms(); char *res = NULL;
        if (j) {
            id = cJSON_GetObjectItem(j, "id");
            cJSON *t = cJSON_GetObjectItem(j, "text"), *w = cJSON_GetObjectItem(j, "words");
            if (cJSON_IsString(t)) res = arctag_tag_text(h, t->valuestring);
            else if (cJSON_IsArray(w)) { char *ws = cJSON_PrintUnformatted(w); res = arctag_tag_words(h, ws); free(ws); }
        } else res = arctag_tag_text(h, line);
        double ms = now_ms() - t0; total += ms;
        cJSON *o = cJSON_CreateObject();
        if (id) cJSON_AddItemToObject(o, "id", cJSON_Duplicate(id, 1)); else cJSON_AddNumberToObject(o, "id", (double)docs);
        cJSON_AddNumberToObject(o, "ms", (double)(long)(ms * 100) / 100);
        cJSON *r = res ? cJSON_Parse(res) : NULL;
        if (r) { words += cJSON_GetObjectItem(r, "n_words")->valueint; cJSON_AddItemToObject(o, "result", r); }
        else cJSON_AddStringToObject(o, "error", arctag_last_error(h));
        char *s = cJSON_PrintUnformatted(o); puts(s); free(s);
        cJSON_Delete(o); cJSON_Delete(j); arctag_free_str(res); free(line);
        docs++;
    }
    if (f != stdin) fclose(f);
    fprintf(stderr, "batch_tag: %ld documents, %ld words, %.1f ms (%.2f ms/word)\n", docs, words, total, words ? total / words : 0.0);
    arctag_destroy(h);
    return 0;
}
