/* Streaming gates through the public API. usage: test_stream MODEL_DIR reference.json
 *   1. Invariance: word-by-word push + poll ends with the same spans, utterances and events as arctag_tag_text, on every reference
 *      text and on longer texts inside the window, with and without timings and speakers.
 *   2. Polling cadence: pushing in batches, or one giant backlog, gives the same result as polling every word.
 *   3. Revisions at several depths, undone again, end in the same output as a stream that never saw them; the last-3-word path used
 *      by ASR interims; revision past the held words is an error that changes nothing.
 *   4. Speakers: labels from word 0 == whole text, without replay; labels arriving late do not change earlier rows, and the size of
 *      the difference from a whole-session flag is reported.
 *   5. Beyond the window: the stream stays within 1 word of the invariants the docs state (row counts, global indices).
 */
#include "arctag.h"
#include "cJSON.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOL 1e-3
static double maxd = 0; static int fails = 0;
static char *slurp(const char *p) {
    FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1); if (fread(b, 1, (size_t)n, f) != (size_t)n) exit(2); b[n] = 0; fclose(f); return b;
}
/* any JSON: same shape and strings, numbers within tol */
static void jcmp(cJSON *a, cJSON *b, double tol, const char *path) {
    if (!a || !b) { if (a != b) { fails++; fprintf(stderr, "presence differs at %s\n", path); } return; }
    if ((a->type & 0xff) != (b->type & 0xff)) { fails++; fprintf(stderr, "type differs at %s\n", path); return; }
    if (cJSON_IsNumber(a)) { double d = fabs(a->valuedouble - b->valuedouble); if (d > maxd) maxd = d; if (d > tol) { fails++; fprintf(stderr, "%s: %f vs %f\n", path, a->valuedouble, b->valuedouble); } return; }
    if (cJSON_IsString(a)) { if (strcmp(a->valuestring, b->valuestring)) { fails++; fprintf(stderr, "%s: %s vs %s\n", path, a->valuestring, b->valuestring); } return; }
    if (cJSON_IsArray(a) || cJSON_IsObject(a)) {
        cJSON *x = a->child, *y = b->child;
        for (; x && y; x = x->next, y = y->next) jcmp(x, y, tol, x->string ? x->string : path);
        if (x || y) { fails++; fprintf(stderr, "length differs at %s\n", path); }
    }
}
static int ends(const char *w) { size_t n = strlen(w); while (n && strchr("\"')]}", w[n - 1])) n--; return n && strchr(".?!", w[n - 1]); }

typedef struct { const char *w; double t0, t1, conf; const char *spk; int final; } tw;
static cJSON *word_obj(const tw *x) {
    cJSON *o = cJSON_CreateObject(); cJSON_AddStringToObject(o, "w", x->w);
    if (x->t0 >= 0) { cJSON_AddNumberToObject(o, "t0", x->t0); cJSON_AddNumberToObject(o, "t1", x->t1); }
    if (x->conf >= 0) cJSON_AddNumberToObject(o, "conf", x->conf);
    if (x->spk) cJSON_AddStringToObject(o, "spk", x->spk);
    if (!x->final) cJSON_AddBoolToObject(o, "final", 0);
    if (ends(x->w)) cJSON_AddBoolToObject(o, "eou", 1);
    return o;
}
static void push(arctag_t *h, const tw *ws, int a, int b, int from) {
    cJSON *q = cJSON_CreateObject(), *arr = cJSON_AddArrayToObject(q, "words");
    if (from >= 0) cJSON_AddNumberToObject(q, "from", from);
    for (int i = a; i < b; i++) cJSON_AddItemToArray(arr, word_obj(ws + i));
    char *js = cJSON_PrintUnformatted(q); cJSON_Delete(q);
    if (arctag_push_words(h, js)) { fails++; fprintf(stderr, "push failed: %s\n", arctag_last_error(h)); }
    free(js);
}
static cJSON *poll_(arctag_t *h) {
    char *r = arctag_poll(h); if (!r) { fprintf(stderr, "poll failed: %s\n", arctag_last_error(h)); exit(1); }
    cJSON *p = cJSON_Parse(r); arctag_free_str(r); return p;
}
static void absorb(cJSON *p, cJSON *ev, cJSON *ut) {
    cJSON *x;
    while ((x = cJSON_DetachItemFromArray(cJSON_GetObjectItem(p, "events"), 0))) cJSON_AddItemToArray(ev, x);
    while ((x = cJSON_DetachItemFromArray(cJSON_GetObjectItem(p, "utterances"), 0))) cJSON_AddItemToArray(ut, x);
}
/* poll every `every` words, then compare to the whole-text result: spans in the last poll's range, all events, closed utterances */
static int polls = 0;
static void stream_vs_whole(arctag_t *h, const tw *ws, int n, int every, cJSON *want, const char *name) {
    cJSON *ev = cJSON_CreateArray(), *ut = cJSON_CreateArray(), *last = NULL;
    arctag_reset(h);
    for (int i = 0; i < n; i += every) {
        int b = i + every < n ? i + every : n;
        push(h, ws, i, b, -1);
        cJSON *p = poll_(h); polls++; absorb(p, ev, ut); cJSON_Delete(last); last = p;
    }
    /* the last poll's spans: those ending after a0 */
    int a0 = cJSON_GetObjectItem(last, "a0")->valueint;
    cJSON *ws2 = cJSON_CreateArray();
    for (cJSON *y = cJSON_GetObjectItem(want, "spans")->child; y; y = y->next) if (cJSON_GetObjectItem(y, "end")->valueint > a0) cJSON_AddItemToArray(ws2, cJSON_Duplicate(y, 1));
    jcmp(cJSON_GetObjectItem(last, "spans"), ws2, TOL, name);
    jcmp(ev, cJSON_GetObjectItem(want, "events"), TOL, name);
    int nu = cJSON_GetArraySize(ut); cJSON *ua = cJSON_CreateArray();
    { int i = 0; for (cJSON *y = cJSON_GetObjectItem(want, "utterances")->child; y && i < nu; y = y->next, i++) cJSON_AddItemToArray(ua, cJSON_Duplicate(y, 1)); }
    for (cJSON *y = ut->child; y; y = y->next) cJSON_DeleteItemFromObject(y, "final");
    jcmp(ut, ua, TOL, name);
    cJSON_Delete(ws2); cJSON_Delete(ua); cJSON_Delete(ev); cJSON_Delete(ut); cJSON_Delete(last);
}
static cJSON *whole(arctag_t *h, const tw *ws, int n) {
    cJSON *a = cJSON_CreateArray();
    for (int i = 0; i < n; i++) cJSON_AddItemToArray(a, word_obj(ws + i));
    char *js = cJSON_PrintUnformatted(a); cJSON_Delete(a);
    char *o = arctag_tag_words(h, js); free(js);
    if (!o) { fprintf(stderr, "tag_words failed: %s\n", arctag_last_error(h)); exit(1); }
    cJSON *w = cJSON_Parse(o); arctag_free_str(o); return w;
}
static tw *plain(char **w, int n, int mode, int spk_from) {
    tw *t = calloc((size_t)n, sizeof(tw));
    for (int i = 0; i < n; i++) {
        t[i].w = w[i]; t[i].final = 1; t[i].t0 = t[i].t1 = t[i].conf = -1;
        if (mode >= 1) { t[i].t0 = 0.31 * i; t[i].t1 = 0.31 * i + 0.25; t[i].conf = 0.5 + 0.01 * (i % 40); t[i].final = i % 5 != 0; }
        if (mode >= 2 && i >= spk_from) t[i].spk = (i / 7) % 2 ? "B" : "A";
    }
    return t;
}
int main(int c, char **v) {
    if (c < 3) { fprintf(stderr, "usage: %s MODEL_DIR reference.json\n", v[0]); return 2; }
    char err[512]; arctag_t *h = arctag_create(v[1], err, sizeof err); if (!h) { puts(err); return 1; }
    cJSON *ref = cJSON_Parse(slurp(v[2])); int nc = 0;
    /* a long text out of the reference texts */
    char *all = malloc(1 << 20); all[0] = 0; int k5 = 0;
    for (cJSON *cs = cJSON_GetObjectItem(ref, "cases")->child; cs && k5 < 5; cs = cs->next, k5++) { strcat(all, cJSON_GetObjectItem(cs, "text")->valuestring); strcat(all, " "); }
    static char *base[8192]; int nb = 0;
    for (char *x = strtok(all, " \n"); x; x = strtok(NULL, " \n")) base[nb++] = x;
    static char *lw[8192]; for (int i = 0; i < 8192; i++) lw[i] = base[i % nb];

    /* ---- 1. invariance */
    for (cJSON *cs = cJSON_GetObjectItem(ref, "cases")->child; cs; cs = cs->next, nc++) {
        arctag_set_option(h, "domain", cJSON_GetObjectItem(cs, "domain")->valuestring);
        char *t = strdup(cJSON_GetObjectItem(cs, "text")->valuestring); static char *w[4096]; int n = 0;
        for (char *x = strtok(t, " "); x; x = strtok(NULL, " ")) w[n++] = x;
        tw *ws = plain(w, n, 0, 0);
        cJSON *want = whole(h, ws, n);
        stream_vs_whole(h, ws, n, 1, want, "reference text");
        cJSON_Delete(want); free(ws); free(t);
    }
    arctag_set_option(h, "domain", "meeting");
    int lens[] = {64, 150, 255, 256}, modes[] = {0, 1, 2};
    for (int l = 0; l < 4; l++) for (int m = 0; m < 3; m++) {
        tw *ws = plain(lw, lens[l], modes[m], 0);
        cJSON *want = whole(h, ws, lens[l]);
        stream_vs_whole(h, ws, lens[l], 1, want, "long text");
        cJSON_Delete(want); free(ws);
    }
    printf("gate 1, word by word vs whole text: %d reference cases + 12 long (up to 256 words; plain, timed, speakers), %d polls, %d mismatches, max score diff %.2e (tol %.0e)\n", nc, polls, fails, maxd, TOL);

    /* ---- 2. polling cadence */
    int f0 = fails; maxd = 0;
    { tw *ws = plain(lw, 250, 2, 0); cJSON *want = whole(h, ws, 250);
      int every[] = {3, 16, 17, 64, 65, 200, 250};
      for (int e = 0; e < 7; e++) stream_vs_whole(h, ws, 250, every[e], want, "cadence");
      cJSON_Delete(want); free(ws); }
    printf("gate 2, poll cadence 3..250 words vs whole text: %d mismatches, max score diff %.2e\n", fails - f0, maxd);

    /* ---- 3. revisions */
    f0 = fails; maxd = 0;
    { int n = 480, nsp = 0; tw *ws = plain(lw, n, 1, 0);
      arctag_t *fresh = arctag_create(v[1], err, sizeof err); if (!fresh) { puts(err); return 1; }
      arctag_set_option(fresh, "domain", "meeting");
      arctag_reset(h);
      for (int i = 0; i < n; i += 40) { int b = i + 40 < n ? i + 40 : n; push(h, ws, i, b, -1); push(fresh, ws, i, b, -1); cJSON_Delete(poll_(h)); cJSON_Delete(poll_(fresh)); }
      int depths[] = {1, 3, 20, 47, 60, 100, 240};
      static tw junk[512]; for (int i = 0; i < 512; i++) { junk[i].w = "xxx"; junk[i].final = 1; junk[i].t0 = junk[i].t1 = junk[i].conf = -1; }
      for (int d = 0; d < 7; d++) {
          int from = n - depths[d];
          push(h, junk, 0, depths[d], from); cJSON *mid = poll_(h);
          if (cJSON_GetObjectItem(mid, "n_words")->valueint != n) { fails++; fprintf(stderr, "revision: n_words\n"); }
          cJSON_Delete(mid);
          push(h, ws, from, n, from);
          cJSON *pa = poll_(h), *pb = poll_(fresh);
          jcmp(cJSON_GetObjectItem(pa, "spans"), cJSON_GetObjectItem(pb, "spans"), TOL, "rev spans");
          jcmp(cJSON_GetObjectItem(pa, "current"), cJSON_GetObjectItem(pb, "current"), TOL, "rev current");
          nsp += cJSON_GetArraySize(cJSON_GetObjectItem(pa, "spans")); cJSON_Delete(pa); cJSON_Delete(pb);
      }
      /* the ASR path: the last 3 words replaced, hundreds of times */
      for (int r = 0; r < 200; r++) { push(h, ws, n - 3, n, n - 3); cJSON_Delete(poll_(h)); }
      { cJSON *pa = poll_(h), *pb = poll_(fresh);
        jcmp(cJSON_GetObjectItem(pa, "spans"), cJSON_GetObjectItem(pb, "spans"), TOL, "rev3 spans");
        jcmp(cJSON_GetObjectItem(pa, "current"), cJSON_GetObjectItem(pb, "current"), TOL, "rev3 current");
        cJSON_Delete(pa); cJSON_Delete(pb); }
      printf("gate 3a, revisions undone at depths 1..240 and 200 x last-3-words, within the fully held session: %d spans compared, %d mismatches, max score diff %.2e\n", nsp, fails - f0, maxd);
      arctag_destroy(fresh); free(ws);

      /* a revision older than the held words: an error, and the stream is unchanged */
      int f1 = fails; int N = 3000; tw *lg = plain(lw, N, 1, 0);
      arctag_reset(h);
      for (int i = 0; i < N; i += 100) { push(h, lg, i, i + 100 < N ? i + 100 : N, -1); cJSON_Delete(poll_(h)); }
      char js[128]; snprintf(js, sizeof js, "{\"from\":5,\"words\":[\"x\"]}");
      if (arctag_push_words(h, js) != -1 || !strstr(arctag_last_error(h), "no longer held")) { fails++; fprintf(stderr, "revision past the held words was not refused\n"); }
      cJSON *pz = poll_(h);
      if (cJSON_GetObjectItem(pz, "n_words")->valueint != N) { fails++; fprintf(stderr, "refused revision changed the stream\n"); }
      cJSON_Delete(pz);
      /* global indices after the window has moved; a deep revision that is still accepted works */
      push(h, junk, 0, 200, N - 200); cJSON *pd = poll_(h);
      if (cJSON_GetObjectItem(pd, "n_words")->valueint != N) { fails++; fprintf(stderr, "deep revision n_words\n"); }
      for (cJSON *sp = cJSON_GetObjectItem(pd, "spans")->child; sp; sp = sp->next) if (cJSON_GetObjectItem(sp, "end")->valueint <= N - 96) { fails++; fprintf(stderr, "deep revision: stale span\n"); }
      cJSON_Delete(pd);
      printf("gate 3b, revision older than the held words refused, stream unchanged; deep replay after the window moved: %d mismatches\n", fails - f1);
      free(lg); }

    /* ---- 4. speakers. Labels from the first word == whole text: gate 1. Labels first appearing mid-session: no replay, so what
     * was reported for earlier words stands, exactly. How far that is from the whole-session flag of tag_words is printed. */
    { int n = 120, at = 60, f2 = fails; tw *ws = plain(lw, n, 2, at);
      arctag_reset(h);
      push(h, ws, 0, at, -1); cJSON *p0 = poll_(h);
      push(h, ws, at, n, -1); cJSON *p1 = poll_(h);
      int a1 = cJSON_GetObjectItem(p1, "a0")->valueint, same = 0;
      /* spans over earlier words, still in the window: the same in both polls */
      cJSON *e0 = cJSON_CreateArray(), *e1 = cJSON_CreateArray();
      for (cJSON *y = cJSON_GetObjectItem(p0, "spans")->child; y; y = y->next) { int e = cJSON_GetObjectItem(y, "end")->valueint; if (e > a1 && e < at) { cJSON_AddItemToArray(e0, cJSON_Duplicate(y, 1)); same++; } }
      for (cJSON *y = cJSON_GetObjectItem(p1, "spans")->child; y; y = y->next) { int e = cJSON_GetObjectItem(y, "end")->valueint; if (e > a1 && e < at) cJSON_AddItemToArray(e1, cJSON_Duplicate(y, 1)); }
      jcmp(e0, e1, 0, "pre-label spans");
      /* against the whole-session flag */
      cJSON *want = whole(h, ws, n); double m0 = maxd; maxd = 0; int fsave = fails, nsame = 0, nw = 0, ng = cJSON_GetArraySize(cJSON_GetObjectItem(p1, "spans"));
      for (cJSON *y = cJSON_GetObjectItem(want, "spans")->child; y; y = y->next) {
          if (cJSON_GetObjectItem(y, "end")->valueint <= a1) continue;
          nw++;
          for (cJSON *g = cJSON_GetObjectItem(p1, "spans")->child; g; g = g->next)
              if (!strcmp(cJSON_GetObjectItem(y, "type")->valuestring, cJSON_GetObjectItem(g, "type")->valuestring) && cJSON_GetObjectItem(y, "start")->valueint == cJSON_GetObjectItem(g, "start")->valueint && cJSON_GetObjectItem(y, "end")->valueint == cJSON_GetObjectItem(g, "end")->valueint) {
                  double d = fabs(cJSON_GetObjectItem(y, "score")->valuedouble - cJSON_GetObjectItem(g, "score")->valuedouble); if (d > maxd) maxd = d; nsame++; break;
              }
      }
      printf("gate 4, a speaker label first appearing at word %d of %d: no replay; %d earlier spans unchanged, %d mismatches. Against tag_words' whole-session flag (differs by design): stream %d spans, whole %d, %d identical in type and range, max score diff %.3f\n", at, n, same, fails - f2, ng, nw, nsame, maxd);
      fails = fsave; maxd = m0;
      cJSON_Delete(e0); cJSON_Delete(e1); cJSON_Delete(p0); cJSON_Delete(p1); cJSON_Delete(want); free(ws); }

    /* a revision that removes the only labels turns the flag back off: the same as a stream that never had them */
    { int n = 40, f5 = fails; tw *pl = plain(lw, n, 0, 0), *lab = plain(lw, n, 0, 0);
      for (int i = n - 5; i < n; i++) lab[i].spk = "A";
      arctag_t *other = arctag_create(v[1], err, sizeof err); arctag_set_option(other, "domain", "meeting");
      arctag_reset(h); push(h, lab, 0, n, -1); cJSON_Delete(poll_(h)); push(h, pl, n - 5, n, n - 5); cJSON *pa = poll_(h);
      push(other, pl, 0, n, -1); cJSON *pb = poll_(other);
      jcmp(cJSON_GetObjectItem(pa, "spans"), cJSON_GetObjectItem(pb, "spans"), TOL, "label revised away");
      jcmp(cJSON_GetObjectItem(pa, "current"), cJSON_GetObjectItem(pb, "current"), TOL, "label revised away");
      printf("gate 4b, the only speaker labels revised away == never labelled: %d mismatches\n", fails - f5);
      cJSON_Delete(pa); cJSON_Delete(pb); arctag_destroy(other); free(pl); free(lab); }

    /* ---- 5. options */
    { int f4 = fails;
      if (arctag_set_option(h, "threads", "8") != 0) { fails++; fprintf(stderr, "threads option refused\n"); }
      if (arctag_set_option(h, "stream_ctx", "5") != -1) { fails++; fprintf(stderr, "bad stream_ctx accepted\n"); }
      int n = 100; tw *ws = plain(lw, n, 0, 0);
      arctag_reset(h); push(h, ws, 0, n, -1); cJSON_Delete(poll_(h));
      if (arctag_set_option(h, "stream_ctx", "64")) { fails++; fprintf(stderr, "stream_ctx refused\n"); }
      cJSON *a = poll_(h);                                /* replays the held words with the new window; no repeated events */
      if (cJSON_GetArraySize(cJSON_GetObjectItem(a, "events"))) { fails++; fprintf(stderr, "stream_ctx change repeated events\n"); }
      cJSON_Delete(a);
      arctag_set_option(h, "stream_ctx", "256"); free(ws);
      printf("gate 5, options: %d mismatches\n", fails - f4); }

    cJSON_Delete(ref); arctag_destroy(h); free(all);
    printf("test_stream: %d mismatches\n", fails);
    return fails ? 1 : 0;
}
