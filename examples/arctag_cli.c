/* arctag_cli: tag a text (argument or stdin) and print the JSON result.
 *   arctag_cli [-m MODEL_DIR] [-d DOMAIN] [-a ASR] [-t THREADS] [--stream] "text ..."
 *   echo "text" | arctag_cli -m model
 * --stream feeds the words one by one through arctag_push_words / arctag_poll (endpoint after . ? !) and prints every poll. */
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

int main(int argc, char **argv) {
    const char *model = getenv("ARCTAG_MODEL_DIR"), *domain = NULL, *asr = NULL, *threads = NULL;
    int stream = 0; char *text = NULL;
    if (!model) model = "model";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) domain = argv[++i];
        else if (!strcmp(argv[i], "-a") && i + 1 < argc) asr = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = argv[++i];
        else if (!strcmp(argv[i], "--stream")) stream = 1;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("usage: %s [-m MODEL_DIR] [-d DOMAIN] [-a ASR] [-t THREADS] [--stream] [TEXT]\n(arctag %s)\n", argv[0], arctag_version());
            return 0;
        } else text = argv[i];
    }
    char err[512];
    arctag_t *h = arctag_create(model, err, sizeof err);
    if (!h) { fprintf(stderr, "arctag_create(%s): %s\n", model, err); return 1; }
    if ((domain && arctag_set_option(h, "domain", domain)) || (asr && arctag_set_option(h, "asr", asr)) ||
        (threads && arctag_set_option(h, "threads", threads))) {
        fprintf(stderr, "option: %s\n", arctag_last_error(h)); arctag_destroy(h); return 1;
    }
    char *owned = NULL;
    if (!text) text = owned = read_stdin();
    if (!stream) {
        char *out = arctag_tag_text(h, text);
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
