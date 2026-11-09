/* A damaged model must produce an error, never a crash. Run under ASan + UBSan by scripts/test_native.sh.
 * usage: test_corrupt MODEL_DIR TMP_DIR ITERATIONS
 *   1. Files: missing, empty, directory, every truncation point class, padded, wrong magic / version / byte order, flipped payload bit.
 *   2. Header: every bit of the header flipped.
 *   3. Table fuzz, with checksums made valid again so the deeper checks are what is tested: every field of every entry set to hostile
 *      values, then random bytes. A file that still loads must run: its forward pass runs under the sanitizers.
 *   4. Index tables: values set out of range.
 *   5. arctag_meta.json and tokenizer.json: truncated, mutated, wrong-checkpoint columns. */
#include "arctag.h"
#include "atn.h"
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define HDR 64
#define ENT 88
static uint32_t crc32_(const unsigned char *p, size_t n) {
    uint32_t c = ~0u;
    while (n--) { c ^= *p++; for (int k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1; }
    return ~c;
}
static uint64_t rng_s = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s; }
static unsigned char *base; static size_t base_n; static unsigned char *scratch;
static long rejected = 0, accepted = 0, ran = 0; static int bad = 0;
static void refresh_header(unsigned char *b) { uint32_t c = crc32_(b, 60); memcpy(b + 60, &c, 4); }
static void run_forward(atn_model *m) {
    /* every row and column the file claims is touched by words of unusual shape */
    atn *a = atn_new(m, atn_num_domains(m) - 1, 64 + ATN_REWIND_WORDS, 64);
    if (!a) return;
    float *o = malloc(sizeof(float) * 40 * (size_t)atn_out_total(m));
    if (o) {
        atn_word ws[40]; memset(ws, 0, sizeof ws);
        for (int i = 0; i < 40; i++) {
            ws[i].npc = i % 10; for (int k = 0; k < 8; k++) ws[i].pcs[k] = (int)(rnd() >> 40);
            for (int k = 0; k < 5; k++) ws[i].wfl[k] = (int)(rnd() % 64) - 2;
            for (int k = 0; k < 8; k++) ws[i].feats[k] = (int)(rnd() % 64) - 2;
            ws[i].src = (int)(rnd() % 9) - 1; ws[i].spk = (int)(rnd() % 16) - 2; ws[i].ust = i % 5 == 0;
        }
        atn_steps(a, ws, 20, o); atn_step(a, ws + 20, o); atn_steps(a, ws + 21, 19, o); ran++;
    }
    free(o); atn_free(a);
}
/* scratch holds a candidate: only its table was changed, so skip the slow payload checksum */
static void try_scratch(void) {
    char err[512]; err[0] = 0;
    atn_model *m = atn_open_mem(scratch, base_n, 0, err, sizeof err);
    if (!m) { rejected++; if (!err[0]) { fprintf(stderr, "rejected without a message\n"); bad++; } return; }
    accepted++; run_forward(m); atn_close(m);
}
static int table_n(void) { uint32_t n; memcpy(&n, base + 12, 4); return (int)n; }
static void write_file(const char *path, const void *p, size_t n) {
    FILE *f = fopen(path, "wb"); if (!f || (n && fwrite(p, 1, n, f) != n)) { fprintf(stderr, "cannot write %s\n", path); exit(2); } fclose(f);
}
static void expect_reject(const char *what, const char *path, const char *needle) {
    char err[512]; err[0] = 0;
    atn_model *m = atn_open(path, err, sizeof err);
    if (m) { fprintf(stderr, "%s: accepted\n", what); bad++; atn_close(m); return; }
    if (!err[0] || (needle && !strstr(err, needle))) { fprintf(stderr, "%s: message '%s' lacks '%s'\n", what, err, needle ? needle : ""); bad++; }
    rejected++;
}
static char *slurp(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(2); }
    fseek(f, 0, SEEK_END); long z = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)z + 1); if (fread(b, 1, (size_t)z, f) != (size_t)z) exit(2); b[z] = 0; fclose(f); if (n) *n = (size_t)z; return b;
}
int main(int c, char **v) {
    if (c < 4) { fprintf(stderr, "usage: %s MODEL_DIR TMP_DIR ITERATIONS\n", v[0]); return 2; }
    const char *md = v[1], *tmp = v[2]; long iters = atol(v[3]);
    char path[1024], p2[1024];
    snprintf(path, sizeof path, "%s/arctag.atw", md);
    { char *t = slurp(path, &base_n); base = aligned_alloc(4096, (base_n + 4095) / 4096 * 4096); memcpy(base, t, base_n); free(t);
      scratch = aligned_alloc(4096, (base_n + 4095) / 4096 * 4096); }
    mkdir(tmp, 0755);
    char f[1024]; snprintf(f, sizeof f, "%s/x.atw", tmp);
    /* ---- 1. files */
    { char err[512]; atn_model *m = atn_open(path, err, sizeof err); if (!m) { fprintf(stderr, "intact file refused: %s\n", err); return 1; } atn_close(m); }
    expect_reject("missing", "/nonexistent/x.atw", "cannot open");
    expect_reject("directory", tmp, "not a weights file");
    write_file(f, base, 0); expect_reject("empty", f, "truncated");
    size_t cuts[] = {1, 3, 4, 63, 64, 65, 100, HDR + ENT, 4095, 4096, 4097, 1 << 16, base_n / 2, base_n - 4096, base_n - 64, base_n - 1};
    for (size_t i = 0; i < sizeof cuts / sizeof *cuts; i++) { write_file(f, base, cuts[i]); expect_reject("truncated", f, cuts[i] < 64 ? "truncated" : "size"); }
    for (long i = 0; i < 40; i++) { size_t cut = rnd() % base_n; write_file(f, base, cut); expect_reject("truncated (random)", f, NULL); }
    { unsigned char *pad = malloc(base_n + 1); memcpy(pad, base, base_n); pad[base_n] = 0; write_file(f, pad, base_n + 1); expect_reject("padded", f, "size"); free(pad); }
    memcpy(scratch, base, base_n); memcpy(scratch, "XXXX", 4); write_file(f, scratch, base_n); expect_reject("magic", f, "magic");
    memcpy(scratch, base, base_n); memcpy(scratch, "ATW1", 4); write_file(f, scratch, base_n); expect_reject("old format", f, "re-export");
    memcpy(scratch, base, base_n); { uint32_t x = 3; memcpy(scratch + 4, &x, 4); } refresh_header(scratch); write_file(f, scratch, base_n); expect_reject("version", f, "version");
    memcpy(scratch, base, base_n); { uint32_t x = 0x04030201; memcpy(scratch + 8, &x, 4); } refresh_header(scratch); write_file(f, scratch, base_n); expect_reject("byte order", f, "byte order");
    /* a flipped bit anywhere in the payload: the checksum catches it, through the file path and the memory path */
    for (int i = 0; i < 8; i++) {
        memcpy(scratch, base, base_n); size_t at = HDR + rnd() % (base_n - HDR); scratch[at] ^= (unsigned char)(1u << (rnd() % 8));
        write_file(f, scratch, base_n); expect_reject("payload bit", f, "checksum");
    }
    for (long i = 0; i < iters / 50 + 8; i++) {
        memcpy(scratch, base, base_n); size_t at = HDR + rnd() % (base_n - HDR); scratch[at] ^= (unsigned char)(1u << (rnd() % 8));
        char err[256]; atn_model *m = atn_open_mem(scratch, base_n, 1, err, sizeof err);
        if (m) { fprintf(stderr, "payload bit %zu: accepted\n", at); bad++; atn_close(m); } else rejected++;
    }
    long file_rej = rejected;
    printf("corrupt 1, files: %ld damaged files refused with a message, %d wrong\n", file_rej, bad);
    /* ---- 2. every header bit */
    memcpy(scratch, base, base_n); long r0 = rejected;
    for (int bit = 0; bit < HDR * 8; bit++) {
        scratch[bit / 8] ^= (unsigned char)(1u << (bit % 8));
        char err[256]; atn_model *m = atn_open_mem(scratch, base_n, 0, err, sizeof err);
        /* bits of the payload crc field are covered by the header crc too; the reserved and sha bytes as well */
        if (m) { fprintf(stderr, "header bit %d: accepted\n", bit); bad++; atn_close(m); } else rejected++;
        scratch[bit / 8] ^= (unsigned char)(1u << (bit % 8));
    }
    printf("corrupt 2, header: %ld bit flips refused, %d wrong\n", rejected - r0, bad);
    /* ---- 3. table: valid checksums, hostile contents */
    r0 = rejected; long a0 = accepted;
    int nt = table_n();
    static const int32_t hostile[] = {-1, 0, 1, 2, 3, 7, 255, 4096, 65536, 0x7FFFFFFF, (int32_t)0x80000000};
    static const int64_t hostile64[] = {-1, 0, 1, 2, 4, 63, 64, 4096, 1ll << 20, 1ll << 40, 0x7FFFFFFFFFFFFFFFll, (int64_t)0x8000000000000000ull};
    memcpy(scratch, base, base_n);
    for (int e = 0; e < nt; e++) {
        unsigned char *ep = scratch + HDR + (size_t)e * ENT;
        unsigned char keep[ENT]; memcpy(keep, ep, ENT);
        for (int fld = 0; fld < 6; fld++) for (size_t h = 0; h < sizeof hostile / sizeof *hostile; h++) {
            memcpy(ep + 48 + 4 * fld, &hostile[h], 4); try_scratch(); memcpy(ep, keep, ENT);
        }
        for (int fld = 0; fld < 2; fld++) for (size_t h = 0; h < sizeof hostile64 / sizeof *hostile64; h++) {
            memcpy(ep + 72 + 8 * fld, &hostile64[h], 8); try_scratch(); memcpy(ep, keep, ENT);
        }
        ep[0] ^= 0x20; try_scratch(); memcpy(ep, keep, ENT);        /* renamed: missing tensor, unsorted table */
        memset(ep, 'a', 48); try_scratch(); memcpy(ep, keep, ENT);  /* no terminator */
    }
    size_t table_bytes = HDR + (size_t)nt * ENT;
    for (long i = 0; i < iters; i++) {                               /* random damage, and a checksum that says it is fine */
        int flips = 1 + (int)(rnd() % 4); unsigned char kept[4][1]; size_t at[4];
        for (int k = 0; k < flips; k++) { at[k] = rnd() % (table_bytes - HDR) + HDR; kept[k][0] = scratch[at[k]]; scratch[at[k]] = (unsigned char)rnd(); }
        try_scratch();
        for (int k = flips - 1; k >= 0; k--) scratch[at[k]] = kept[k][0];
    }
    /* header counts too, with a valid header checksum */
    for (long i = 0; i < iters / 10; i++) {
        unsigned char kh[HDR]; memcpy(kh, scratch, HDR);
        int fld = 8 + 4 * (int)(rnd() % 6); uint32_t val = (uint32_t)rnd(); if (rnd() & 1) val &= 0xFFF;
        memcpy(scratch + fld, &val, 4); refresh_header(scratch);
        try_scratch();
        memcpy(scratch, kh, HDR);
    }
    printf("corrupt 3, table: %ld refused, %ld still loadable and run through the forward pass under sanitizers (%ld runs), %d wrong\n", rejected - r0, accepted - a0, ran, bad);
    /* ---- 4. index tables, read-out group tables, span head slices */
    r0 = rejected; a0 = accepted;
    for (int e = 0; e < nt; e++) {
        unsigned char *ep = scratch + HDR + (size_t)e * ENT; const char *nm = (const char *)ep;
        int32_t dt, d0; int64_t off; memcpy(&dt, ep + 48, 4); memcpy(&d0, ep + 56, 4); memcpy(&off, ep + 72, 8);
        if (dt != 2 || d0 < 1) continue;
        uint64_t blob; memcpy(&blob, scratch + 24, 8);
        int32_t *vals = (int32_t *)(void *)(scratch + blob + off); int32_t keep = vals[0], last = vals[d0 - 1];
        int32_t evil[] = {-1, 1 << 20, 0x7FFFFFFF, (int32_t)0x80000000};
        for (int h = 0; h < 4; h++) {
            vals[0] = evil[h]; { char err[256]; atn_model *m = atn_open_mem(scratch, base_n, 0, err, sizeof err); if (m) { fprintf(stderr, "index table %s[0] = %d: accepted\n", nm, evil[h]); bad++; atn_close(m); } else rejected++; }
            vals[0] = keep;
            vals[d0 - 1] = evil[h]; { char err[256]; atn_model *m = atn_open_mem(scratch, base_n, 0, err, sizeof err); if (m) { fprintf(stderr, "index table %s[last] = %d: accepted\n", nm, evil[h]); bad++; atn_close(m); } else rejected++; }
            vals[d0 - 1] = last;
        }
    }
    printf("corrupt 4, index tables: %ld out-of-range values refused, %d wrong\n", rejected - r0, bad);
    /* ---- 5. meta and tokenizer: the same weights (linked), a damaged companion */
    char dir[1024]; snprintf(dir, sizeof dir, "%s/m", tmp); mkdir(dir, 0755);
    char ap[1024], rp[1024]; snprintf(ap, sizeof ap, "%s/arctag.atw", dir); if (!realpath(path, rp)) { fprintf(stderr, "realpath\n"); return 2; } unlink(ap); if (symlink(rp, ap)) { fprintf(stderr, "symlink\n"); return 2; }
    snprintf(p2, sizeof p2, "%s/arctag_meta.json", md); size_t mn; char *meta = slurp(p2, &mn);
    snprintf(p2, sizeof p2, "%s/tokenizer.json", md); size_t tn; char *tok = slurp(p2, &tn);
    char mp[1024], tp[1024]; snprintf(mp, sizeof mp, "%s/arctag_meta.json", dir); snprintf(tp, sizeof tp, "%s/tokenizer.json", dir);
    write_file(mp, meta, mn); write_file(tp, tok, tn);
    char err[512]; arctag_t *h = arctag_create(dir, err, sizeof err); if (!h) { fprintf(stderr, "intact dir refused: %s\n", err); return 1; } arctag_destroy(h);
    long created = 0, refused = 0;
    char *mut = malloc((mn > tn ? mn : tn) + 1);
    long meta_iters = iters / 20 + 20;
    for (long i = 0; i < 2 * meta_iters; i++) {
        int which = i % 2; const char *src = which ? tok : meta; size_t n = which ? tn : mn;
        memcpy(mut, src, n); size_t L = n;
        switch (rnd() % 4) {
            case 0: L = rnd() % n; break;                                                     /* truncated */
            case 1: for (int k = 0; k < 8; k++) mut[rnd() % n] = (char)rnd(); break;          /* bytes */
            case 2: { size_t a = rnd() % n, b = a + rnd() % 4096; if (b > n) b = n; memmove(mut + a, mut + b, n - b); L = n - (b - a); break; }   /* a hole */
            case 3: { size_t a = rnd() % n; const char *rp2 = rnd() & 1 ? "null" : "[[[1e999,\"x\"]]]"; size_t rl = strlen(rp2); if (a + rl < n) memcpy(mut + a, rp2, rl); break; }
        }
        mut[L] = 0;
        write_file(which ? tp : mp, mut, L);
        h = arctag_create(dir, err, sizeof err);
        if (h) {
            created++;
            char *o = arctag_tag_text(h, "Call Maria at 415 555 0199 tomorrow."); arctag_free_str(o);
            arctag_push_words(h, "[\"a\",{\"w\":\"b\",\"spk\":\"A\"}]"); o = arctag_poll(h); arctag_free_str(o);
            arctag_destroy(h);
        } else { refused++; if (!err[0]) { fprintf(stderr, "refused without a message\n"); bad++; } }
        write_file(which ? tp : mp, src, n);
    }
    /* a meta from another checkpoint: a column too many. Must be refused with a message that says so. */
    { const char *q = strstr(meta, "\"item\""); const char *br = q ? strchr(q, '[') : NULL;
      if (br) { size_t at = (size_t)(br - meta) + 1; size_t w = 0; memcpy(mut, meta, at); w = at; memcpy(mut + w, "\"item.extra\",", 13); w += 13; memcpy(mut + w, meta + at, mn - at); w += mn - at; mut[w] = 0;
        write_file(mp, mut, w); h = arctag_create(dir, err, sizeof err);
        if (h) { fprintf(stderr, "a meta with another column count was accepted\n"); bad++; arctag_destroy(h); }
        else if (!strstr(err, "different checkpoints")) { fprintf(stderr, "wrong message for a mismatched meta: %s\n", err); bad++; }
        write_file(mp, meta, mn); }
      else { fprintf(stderr, "test bug: no item read-out in meta\n"); bad++; } }
    /* no weights file, only an onnx: the message says what to do */
    { unlink(ap); char op[1024]; snprintf(op, sizeof op, "%s/arctag.onnx", dir); write_file(op, "x", 1); h = arctag_create(dir, err, sizeof err);
      if (h || !strstr(err, "export_model")) { fprintf(stderr, "onnx-only dir: '%s'\n", err); bad++; if (h) arctag_destroy(h); } unlink(op); }
    printf("corrupt 5, meta + tokenizer: %ld mutated copies, %ld refused with a message, %ld loaded and ran; %d wrong\n", 2 * meta_iters, refused, created, bad);
    printf("test_corrupt: %ld refused, %ld accepted-and-ran, %d wrong\n", rejected + refused, accepted + created, bad);
    free(mut); free(meta); free(tok); free(base); free(scratch);
    return bad ? 1 : 0;
}
