#include "tokenizer.h"
#include "unicode_tables.h"
#include "cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ unicode */
static int in_ranges(const uint32_t (*r)[2], size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < r[mid][0]) hi = mid;
        else if (cp > r[mid][1]) lo = mid + 1;
        else return 1;
    }
    return 0;
}
#define NR(x) (sizeof(x) / sizeof(x[0]))
int uni_is_letter(uint32_t cp) { return in_ranges(UT_LETTER, NR(UT_LETTER), cp); }
int uni_is_number(uint32_t cp) { return in_ranges(UT_NUMBER, NR(UT_NUMBER), cp); }
int uni_is_space(uint32_t cp) { return in_ranges(UT_SPACE, NR(UT_SPACE), cp); }
int uni_is_digit(uint32_t cp) { return in_ranges(UT_DIGIT, NR(UT_DIGIT), cp); }

int utf8_decode(const char *s, size_t n, size_t *i, uint32_t *cp) {
    const unsigned char *p = (const unsigned char *)s + *i;
    size_t left = n - *i;
    if (!left) return -1;
    unsigned c = p[0];
    int len; uint32_t v;
    if (c < 0x80) { len = 1; v = c; }
    else if ((c & 0xE0) == 0xC0) { len = 2; v = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { len = 3; v = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { len = 4; v = c & 0x07; }
    else { *cp = 0xFFFD; *i += 1; return 0; }                   /* invalid lead byte: consume 1 byte */
    if ((size_t)len > left) { *cp = 0xFFFD; *i += 1; return 0; }
    for (int k = 1; k < len; k++) {
        if ((p[k] & 0xC0) != 0x80) { *cp = 0xFFFD; *i += 1; return 0; }
        v = (v << 6) | (p[k] & 0x3F);
    }
    *cp = v; *i += len;
    return 0;
}

static int utf8_encode(uint32_t cp, char *out) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) { out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) { out[0] = (char)(0xE0 | (cp >> 12)); out[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
    out[0] = (char)(0xF0 | (cp >> 18)); out[1] = (char)(0x80 | ((cp >> 12) & 0x3F)); out[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* ------------------------------------------------------------------ string -> int hash map */
typedef struct { char *key; int32_t val; } hent;
typedef struct { hent *e; size_t cap, n; } hmap;

static uint64_t fnv(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ULL; }
    return h;
}
static int hm_init(hmap *m, size_t want) {
    m->cap = 16; while (m->cap < want * 2) m->cap <<= 1;
    m->e = (hent *)calloc(m->cap, sizeof(hent)); m->n = 0;
    return m->e ? 0 : -1;
}
static int hm_put(hmap *m, const char *k, size_t kn, int32_t v) {
    size_t i = fnv(k, kn) & (m->cap - 1);
    while (m->e[i].key) {
        if (strlen(m->e[i].key) == kn && !memcmp(m->e[i].key, k, kn)) { m->e[i].val = v; return 0; }
        i = (i + 1) & (m->cap - 1);
    }
    char *c = (char *)malloc(kn + 1); if (!c) return -1;
    memcpy(c, k, kn); c[kn] = 0; m->e[i].key = c; m->e[i].val = v; m->n++;
    return 0;
}
static int hm_get(const hmap *m, const char *k, size_t kn, int32_t *v) {
    size_t i = fnv(k, kn) & (m->cap - 1);
    while (m->e[i].key) {
        if (strlen(m->e[i].key) == kn && !memcmp(m->e[i].key, k, kn)) { *v = m->e[i].val; return 1; }
        i = (i + 1) & (m->cap - 1);
    }
    return 0;
}
static void hm_free(hmap *m) {
    if (!m->e) return;
    for (size_t i = 0; i < m->cap; i++) free(m->e[i].key);
    free(m->e); m->e = NULL;
}

/* ------------------------------------------------------------------ tokenizer */
#define MAX_ADDED 64
struct arctok {
    hmap vocab, merges;
    char bytemap[256][5];                  /* byte -> UTF-8 of its GPT-2 byte-level char */
    char *added[MAX_ADDED]; int32_t added_id[MAX_ADDED]; int n_added;
};

static void build_bytemap(arctok_t *t) {
    int used[256] = {0}; int n = 0;
    for (int b = 0; b < 256; b++)
        if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) used[b] = 1;
    for (int b = 0; b < 256; b++) {
        uint32_t cp = used[b] ? (uint32_t)b : (uint32_t)(256 + n++);
        int k = utf8_encode(cp, t->bytemap[b]); t->bytemap[b][k] = 0;
    }
}

static void seterr(char *err, size_t errlen, const char *msg, const char *x) {
    if (err && errlen) snprintf(err, errlen, "%s%s", msg, x ? x : "");
}

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n + 1);
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); b = NULL; }
    fclose(f);
    if (b) { b[n] = 0; if (len) *len = (size_t)n; }
    return b;
}

arctok_t *arctok_load(const char *path, char *err, size_t errlen) {
    size_t n; char *txt = read_file(path, &n);
    if (!txt) { seterr(err, errlen, "cannot read ", path); return NULL; }
    cJSON *j = cJSON_Parse(txt); free(txt);
    if (!j) { seterr(err, errlen, "invalid JSON: ", path); return NULL; }
    arctok_t *t = (arctok_t *)calloc(1, sizeof(arctok_t));
    cJSON *model = cJSON_GetObjectItem(j, "model");
    cJSON *vocab = model ? cJSON_GetObjectItem(model, "vocab") : NULL;
    cJSON *merges = model ? cJSON_GetObjectItem(model, "merges") : NULL;
    cJSON *type = model ? cJSON_GetObjectItem(model, "type") : NULL;
    if (!t || !vocab || !merges || !cJSON_IsString(type) || strcmp(type->valuestring, "BPE")) {
        seterr(err, errlen, "tokenizer.json: expected a BPE model with vocab + merges", NULL);
        cJSON_Delete(j); free(t); return NULL;
    }
    hm_init(&t->vocab, (size_t)cJSON_GetArraySize(vocab));
    hm_init(&t->merges, (size_t)cJSON_GetArraySize(merges));
    for (cJSON *v = vocab->child; v; v = v->next) hm_put(&t->vocab, v->string, strlen(v->string), (int32_t)v->valuedouble);
    int rank = 0; char buf[1024];
    for (cJSON *m = merges->child; m; m = m->next, rank++) {
        const char *a = NULL, *b = NULL;
        if (cJSON_IsArray(m) && cJSON_GetArraySize(m) == 2) { a = cJSON_GetArrayItem(m, 0)->valuestring; b = cJSON_GetArrayItem(m, 1)->valuestring; }
        else if (cJSON_IsString(m)) {                               /* old format "a b" */
            snprintf(buf, sizeof buf, "%s", m->valuestring);
            hm_put(&t->merges, buf, strlen(buf), rank); continue;
        }
        if (!a || !b) continue;
        int k = snprintf(buf, sizeof buf, "%s %s", a, b);
        if (k > 0 && (size_t)k < sizeof buf) hm_put(&t->merges, buf, (size_t)k, rank);
    }
    cJSON *added = cJSON_GetObjectItem(j, "added_tokens");
    for (cJSON *a = added ? added->child : NULL; a && t->n_added < MAX_ADDED; a = a->next) {
        cJSON *c = cJSON_GetObjectItem(a, "content"), *id = cJSON_GetObjectItem(a, "id");
        if (cJSON_IsString(c) && cJSON_IsNumber(id) && c->valuestring[0]) {
            t->added[t->n_added] = strdup(c->valuestring); t->added_id[t->n_added++] = (int32_t)id->valuedouble;
        }
    }
    cJSON_Delete(j);
    build_bytemap(t);
    return t;
}

void arctok_free(arctok_t *t) {
    if (!t) return;
    hm_free(&t->vocab); hm_free(&t->merges);
    for (int i = 0; i < t->n_added; i++) free(t->added[i]);
    free(t);
}

/* BPE of one pre-token (bytes s[0..n)) -> ids appended */
static int bpe_piece(const arctok_t *t, const char *s, size_t n, int32_t *ids, int cap, int k) {
    /* mapped string + symbol boundaries (each symbol = one byte's char initially) */
    size_t mcap = n * 4 + 1;
    char *ms = (char *)malloc(mcap);
    size_t *st = (size_t *)malloc((n + 1) * sizeof(size_t));
    size_t *ln = (size_t *)malloc((n + 1) * sizeof(size_t));
    char *key = (char *)malloc(mcap * 2 + 2);
    if (!ms || !st || !ln || !key) { free(ms); free(st); free(ln); free(key); return -1; }
    size_t m = 0; int ns = 0;
    for (size_t i = 0; i < n; i++) {
        const char *c = t->bytemap[(unsigned char)s[i]]; size_t cl = strlen(c);
        st[ns] = m; ln[ns] = cl; ns++;
        memcpy(ms + m, c, cl); m += cl;
    }
    for (;;) {
        int best = -1; int32_t br = 0;
        for (int i = 0; i + 1 < ns; i++) {
            size_t kl = ln[i] + 1 + ln[i + 1];
            memcpy(key, ms + st[i], ln[i]); key[ln[i]] = ' '; memcpy(key + ln[i] + 1, ms + st[i + 1], ln[i + 1]);
            int32_t r;
            if (hm_get(&t->merges, key, kl, &r) && (best < 0 || r < br)) { best = i; br = r; }
        }
        if (best < 0) break;
        ln[best] += ln[best + 1];                                   /* symbols are contiguous in ms */
        for (int i = best + 1; i + 1 < ns; i++) { st[i] = st[i + 1]; ln[i] = ln[i + 1]; }
        ns--;
    }
    for (int i = 0; i < ns; i++) {
        int32_t id;
        if (!hm_get(&t->vocab, ms + st[i], ln[i], &id)) {           /* byte-level: every byte char is in the vocab */
            for (size_t b = st[i]; b < st[i] + ln[i];) {             /* fall back to single chars */
                size_t b0 = b; uint32_t cp; utf8_decode(ms, st[i] + ln[i], &b, &cp);
                if (hm_get(&t->vocab, ms + b0, b - b0, &id)) { if (k < cap) ids[k] = id; k++; }
            }
            continue;
        }
        if (k < cap) ids[k] = id;
        k++;
    }
    free(ms); free(st); free(ln); free(key);
    return k;
}

typedef enum { C_L, C_N, C_S, C_O } ccls;
static ccls cls_of(uint32_t cp) {
    if (uni_is_letter(cp)) return C_L;
    if (uni_is_number(cp)) return C_N;
    if (uni_is_space(cp)) return C_S;
    return C_O;
}

/* GPT-2 pre-tokenizer: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+ */
static int pretok_bpe(const arctok_t *t, const char *s, size_t n, int32_t *ids, int cap, int k) {
    size_t i = 0;
    /* decode once */
    uint32_t *cp = (uint32_t *)malloc((n + 1) * sizeof(uint32_t));
    size_t *off = (size_t *)malloc((n + 2) * sizeof(size_t));
    if (!cp || !off) { free(cp); free(off); return -1; }
    size_t nc = 0;
    while (i < n) { off[nc] = i; utf8_decode(s, n, &i, &cp[nc]); nc++; }
    off[nc] = n;
    size_t p = 0;
    while (p < nc && k >= 0) {
        size_t e = p;
        uint32_t c = cp[p];
        if (c == '\'' && p + 1 < nc) {
            uint32_t a = cp[p + 1], b = p + 2 < nc ? cp[p + 2] : 0;
            if (a == 's' || a == 't' || a == 'm' || a == 'd') e = p + 2;
            else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) e = p + 3;
        }
        if (e == p) {
            size_t q = p;
            if (c == ' ' && p + 1 < nc && cls_of(cp[p + 1]) != C_S) q = p + 1;
            ccls k0 = q < nc ? cls_of(cp[q]) : C_S;
            if (k0 != C_S && !(q == p && c == ' ')) {
                e = q;
                while (e < nc && cls_of(cp[e]) == k0) e++;
            } else {                                                 /* whitespace run */
                size_t r = p;
                while (r < nc && cls_of(cp[r]) == C_S) r++;
                if (r == nc || r - p == 1) e = r;                     /* \s+(?!\S) at the end, or \s+ (single) */
                else e = r - 1;                                       /* leave the last space for the next token */
            }
        }
        k = bpe_piece(t, s + off[p], off[e] - off[p], ids, cap, k);
        p = e;
    }
    free(cp); free(off);
    return k;
}

int arctok_encode(const arctok_t *t, const char *text, int32_t *ids, int cap) {
    if (!t || !text) return -1;
    size_t n = strlen(text), a = 0; int k = 0;
    while (a <= n && k >= 0) {
        /* earliest added (special) token at or after a */
        size_t best = n; int bi = -1;
        for (int j = 0; j < t->n_added; j++) {
            const char *f = strstr(text + a, t->added[j]);
            if (f && (size_t)(f - text) < best) { best = (size_t)(f - text); bi = j; }
        }
        if (best > a) k = pretok_bpe(t, text + a, best - a, ids, cap, k);
        if (bi < 0) break;
        if (k < cap) ids[k] = t->added_id[bi];
        k++;
        a = best + strlen(t->added[bi]);
    }
    return k;
}
