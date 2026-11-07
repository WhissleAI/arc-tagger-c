/* C tokenizer == HuggingFace tokenizers (expected ids from tests/make_expected.py) */
#include "tokenizer.h"
#include "cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s tokenizer.json tokens.json\n", argv[0]); return 2; }
    char err[256];
    arctok_t *t = arctok_load(argv[1], err, sizeof err);
    if (!t) { fprintf(stderr, "%s\n", err); return 1; }
    FILE *f = fopen(argv[2], "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n + 1); if (fread(b, 1, (size_t)n, f) != (size_t)n) return 1; b[n] = 0; fclose(f);
    cJSON *j = cJSON_Parse(b); free(b);
    int fails = 0, total = 0;
    for (cJSON *c = cJSON_GetObjectItem(j, "cases")->child; c; c = c->next, total++) {
        const char *text = cJSON_GetObjectItem(c, "text")->valuestring;
        cJSON *ids = cJSON_GetObjectItem(c, "ids");
        int32_t got[512]; int k = arctok_encode(t, text, got, 512), want = cJSON_GetArraySize(ids), ok = k == want;
        for (int i = 0; ok && i < k; i++) ok = got[i] == cJSON_GetArrayItem(ids, i)->valueint;
        if (!ok) {
            fails++; fprintf(stderr, "MISMATCH '%s': got", text);
            for (int i = 0; i < k && i < 512; i++) fprintf(stderr, " %d", got[i]);
            fprintf(stderr, " | want");
            for (cJSON *x = ids->child; x; x = x->next) fprintf(stderr, " %d", x->valueint);
            fprintf(stderr, "\n");
        }
    }
    printf("tokenizer: %d/%d cases identical\n", total - fails, total);
    cJSON_Delete(j); arctok_free(t);
    return fails ? 1 : 0;
}
