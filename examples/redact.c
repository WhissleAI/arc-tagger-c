/* redact: PII (and optionally profanity / toxicity) redaction of text, one document per line.
 *   redact [-m MODEL_DIR] [-c CONFIG.json] [--int8] [-d DOMAIN] [--policy POLICY] [--json] [INPUT|-]
 * POLICY: "mask" (default: every PII category except public contacts, plus profanity / toxicity), or a list such as
 * "phone=mask,email=mask,name=mask,card=mask,contact=off" (categories: phone email address card gov_id dob name handle
 * contact profanity toxicity). Prints the redacted line, or with --json {"redacted": ..., "spans": [masked spans]}. */
#include "arctag.h"
#include "cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_line(FILE *f) {
    size_t cap = 4096, n = 0; char *b = (char *)malloc(cap); int c;
    while ((c = fgetc(f)) != EOF && c != '\n') { if (n + 1 >= cap) { cap *= 2; b = (char *)realloc(b, cap); } b[n++] = (char)c; }
    if (c == EOF && n == 0) { free(b); return NULL; }
    if (n && b[n - 1] == '\r') n--;
    b[n] = 0; return b;
}

int main(int argc, char **argv) {
    const char *model = NULL, *config = NULL, *domain = NULL, *policy = NULL, *in = "-";
    int int8 = 0, json = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) config = argv[++i];
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) domain = argv[++i];
        else if (!strcmp(argv[i], "--policy") && i + 1 < argc) policy = argv[++i];
        else if (!strcmp(argv[i], "--int8")) int8 = 1;
        else if (!strcmp(argv[i], "--json")) json = 1;
        else if (argv[i][0] == '-' && argv[i][1]) { fprintf(stderr, "usage: %s [-m MODEL_DIR] [-c CONFIG.json] [--int8] [-d DOMAIN] [--policy P] [--json] [INPUT|-]\n", argv[0]); return 2; }
        else in = argv[i];
    }
    char err[512];
    arctag_t *h;
    if (config) h = arctag_create_from_config(model, config, err, sizeof err);
    else { arctag_options o; arctag_options_init(&o); o.precision = int8 ? ARCTAG_INT8 : ARCTAG_FP32; h = arctag_create_ex(model ? model : "model", &o, err, sizeof err); }
    if (!h) { fprintf(stderr, "arctag: %s\n", err); return 1; }
    if ((config && int8 && arctag_set_option(h, "precision", "int8")) || (domain && arctag_set_option(h, "domain", domain)) ||
        ((policy || !config) && arctag_set_option(h, "redact", policy ? policy : "mask"))) {
        fprintf(stderr, "option: %s\n", arctag_last_error(h)); arctag_destroy(h); return 1;
    }
    FILE *f = strcmp(in, "-") ? fopen(in, "rb") : stdin;
    if (!f) { fprintf(stderr, "cannot open %s\n", in); arctag_destroy(h); return 1; }
    char *line;
    while ((line = read_line(f)) != NULL) {
        if (!*line) { puts(""); free(line); continue; }
        char *res = arctag_tag_text(h, line);
        cJSON *r = res ? cJSON_Parse(res) : NULL;
        const char *red = r ? cJSON_GetStringValue(cJSON_GetObjectItem(r, "redacted_text")) : NULL;
        if (!red) { fprintf(stderr, "error: %s\n", res ? "redaction is off in this config" : arctag_last_error(h)); }
        else if (!json) puts(red);
        else {
            cJSON *o = cJSON_CreateObject(), *sp = cJSON_AddArrayToObject(o, "spans");
            cJSON_AddStringToObject(o, "redacted", red);
            for (cJSON *s = cJSON_GetObjectItem(r, "spans")->child; s; s = s->next) {
                const char *fam = cJSON_GetStringValue(cJSON_GetObjectItem(s, "family"));
                if (fam && (!strcmp(fam, "pii") || !strcmp(fam, "profanity"))) {
                    cJSON *x = cJSON_CreateObject();
                    cJSON_AddItemToObject(x, "type", cJSON_Duplicate(cJSON_GetObjectItem(s, "type"), 1));
                    cJSON_AddItemToObject(x, "start", cJSON_Duplicate(cJSON_GetObjectItem(s, "start"), 1));
                    cJSON_AddItemToObject(x, "end", cJSON_Duplicate(cJSON_GetObjectItem(s, "end"), 1));
                    cJSON_AddItemToObject(x, "score", cJSON_Duplicate(cJSON_GetObjectItem(s, "score"), 1));
                    cJSON_AddItemToArray(sp, x);
                }
            }
            char *s = cJSON_PrintUnformatted(o); puts(s); free(s); cJSON_Delete(o);
        }
        cJSON_Delete(r); arctag_free_str(res); free(line);
    }
    if (f != stdin) fclose(f);
    arctag_destroy(h);
    return 0;
}
