/* arctag_cli: tag a text (argument or stdin) and print the JSON result.
 *   arctag_cli [-m MODEL_DIR] [-c CONFIG.json] [--int8] [-d DOMAIN] [-a ASR] [-t THREADS] [--redact POLICY] [--emit LIST]
 *              [--tags PREFIXES] [--thr PATTERN=VALUE]... [--words] [--stream] [--info] "text ..."
 *   echo "text" | arctag_cli -m model
 * --words reads a JSON word array (strings or {"w","t0","t1","conf","spk","eou"}) instead of plain text.
 * --stream feeds the words one by one through arctag_push_words / arctag_poll (endpoint after . ? !) and prints every poll.
 * Options on the command line are applied after the config file. */
#include "arctag.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define strtok_r strtok_s
#define strdup _strdup
#endif

static char *read_stdin(void) {
    size_t cap = 4096, n = 0; char *b = (char *)malloc(cap); int c;
    while ((c = getchar()) != EOF) { if (n + 1 >= cap) { cap *= 2; b = (char *)realloc(b, cap); } b[n++] = (char)c; }
    b[n] = 0; return b;
}

static void json_escape(const char *s, char *out, size_t cap) {
    size_t k = 0;
    for (; *s && k + 7 < cap; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { out[k++] = '\\'; out[k++] = (char)c; }
        else if (c < 0x20) k += (size_t)snprintf(out + k, cap - k, "\\u%04x", c);
        else out[k++] = (char)c;
    }
    out[k] = 0;
}

static void usage(const char *p) {
    printf("usage: %s [-m MODEL_DIR] [-c CONFIG.json] [--int8|--fp32] [-d DOMAIN] [-a ASR] [-t THREADS]\n"
           "          [--redact off|mask|cat=mask,...] [--emit SECTIONS] [--tags PREFIXES] [--thr PATTERN=VALUE]...\n"
           "          [--words] [--stream] [--info] [TEXT]\n(arctag %s)\n", p, arctag_version());
}

int main(int argc, char **argv) {
    const char *model = getenv("ARCTAG_MODEL_DIR"), *config = NULL;
    const char *keys[64], *vals[64]; int nkv = 0;
    int stream = 0, info = 0, words = 0, precision = -1; char *text = NULL;
    static char thrkey[64][300];
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(a, "-c") && i + 1 < argc) config = argv[++i];
        else if (!strcmp(a, "--int8")) precision = ARCTAG_INT8;
        else if (!strcmp(a, "--fp32")) precision = ARCTAG_FP32;
        else if (!strcmp(a, "-d") && i + 1 < argc && nkv < 64) { keys[nkv] = "domain"; vals[nkv++] = argv[++i]; }
        else if (!strcmp(a, "-a") && i + 1 < argc && nkv < 64) { keys[nkv] = "asr"; vals[nkv++] = argv[++i]; }
        else if (!strcmp(a, "-t") && i + 1 < argc && nkv < 64) { keys[nkv] = "threads"; vals[nkv++] = argv[++i]; }
        else if (!strcmp(a, "--redact") && i + 1 < argc && nkv < 64) { keys[nkv] = "redact"; vals[nkv++] = argv[++i]; }
        else if (!strcmp(a, "--emit") && i + 1 < argc && nkv < 64) { keys[nkv] = "emit"; vals[nkv++] = argv[++i]; }
        else if (!strcmp(a, "--tags") && i + 1 < argc && nkv < 64) { keys[nkv] = "tags"; vals[nkv++] = argv[++i]; }
        else if (!strcmp(a, "--thr") && i + 1 < argc && nkv < 64) {
            const char *kv = argv[++i], *eq = strrchr(kv, '=');
            if (!eq) { fprintf(stderr, "--thr expects PATTERN=VALUE\n"); return 2; }
            snprintf(thrkey[nkv], sizeof thrkey[nkv], "thr:%.*s", (int)(eq - kv), kv);
            keys[nkv] = thrkey[nkv]; vals[nkv++] = eq + 1;
        }
        else if (!strcmp(a, "--stream")) stream = 1;
        else if (!strcmp(a, "--words")) words = 1;
        else if (!strcmp(a, "--info")) info = 1;
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(argv[0]); return 0; }
        else text = argv[i];
    }
    char err[512];
    arctag_t *h;
    if (config) {
        h = arctag_create_from_config(model, config, err, sizeof err);
        if (h && precision >= 0 && arctag_set_option(h, "precision", precision == ARCTAG_INT8 ? "int8" : "fp32")) {
            fprintf(stderr, "precision: %s\n", arctag_last_error(h)); arctag_destroy(h); return 1;
        }
    } else {
        arctag_options o; arctag_options_init(&o);
        if (precision >= 0) o.precision = (arctag_precision_t)precision;
        h = arctag_create_ex(model ? model : "model", &o, err, sizeof err);
    }
    if (!h) { fprintf(stderr, "arctag_create(%s): %s\n", model ? model : "model", err); return 1; }
    for (int i = 0; i < nkv; i++)
        if (arctag_set_option(h, keys[i], vals[i])) { fprintf(stderr, "option %s: %s\n", keys[i], arctag_last_error(h)); arctag_destroy(h); return 1; }
    if (info) { char *s = arctag_info(h); printf("%s\n", s); arctag_free_str(s); if (!text) { arctag_destroy(h); return 0; } }
    char *owned = NULL;
    if (!text) text = owned = read_stdin();
    if (!stream) {
        char *out = words ? arctag_tag_words(h, text) : arctag_tag_text(h, text);
        if (!out) { fprintf(stderr, "error: %s\n", arctag_last_error(h)); arctag_destroy(h); free(owned); return 1; }
        printf("%s\n", out);
        arctag_free_str(out);
    } else {
        char *copy = strdup(text), *save = NULL, buf[4096], esc[2048];
        for (char *w = strtok_r(copy, " \t\r\n", &save); w; w = strtok_r(NULL, " \t\r\n", &save)) {
            size_t L = strlen(w); int eou = L && strchr(".?!", w[L - 1]) != NULL;
            json_escape(w, esc, sizeof esc);
            snprintf(buf, sizeof buf, "{\"words\":[\"%s\"],\"endpoint\":%s}", esc, eou ? "true" : "false");
            if (arctag_push_words(h, buf)) { fprintf(stderr, "push: %s\n", arctag_last_error(h)); break; }
            char *out = arctag_poll(h);
            if (out) { printf("%s\n", out); arctag_free_str(out); }
        }
        free(copy);
    }
    free(owned);
    arctag_destroy(h);
    return 0;
}
