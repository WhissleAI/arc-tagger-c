/* tagsrv.c: see tagsrv.h. Part of arc-tagger-c (Whissle Inc.), CC BY-NC 4.0 with the Whissle terms; see LICENSE. */
#define _GNU_SOURCE   /* strcasestr, M_PI, strdup, strtok_r, clock_gettime on glibc with -std=c11 */
#include "tagsrv.h"

#include <ctype.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "arctag.h"

static struct {
  char model_dir[1024];
  char model_id[256];
  char domain[64];
  char* config_json;        /* ARCTAG_CONFIG file contents, or NULL */
  arctag_options opt;
  int n;
  arctag_t** h;
  int* busy;
  int* dirty;
  pthread_mutex_t mu;
  pthread_cond_t cv;
  int queue_timeout_s;
  int max_streams;
  atomic_int streams;
  int ready;
} G = {.mu = PTHREAD_MUTEX_INITIALIZER, .cv = PTHREAD_COND_INITIALIZER};

static const char* env_or(const char* k, const char* d) {
  const char* v = getenv(k);
  return v && *v ? v : d;
}

static char* read_file(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char* s = malloc((size_t)n + 1);
  if (fread(s, 1, (size_t)n, f) != (size_t)n) { free(s); fclose(f); return NULL; }
  s[n] = 0;
  fclose(f);
  return s;
}

static cJSON* err_obj(const char* fmt, const char* arg) {
  char m[1024];
  snprintf(m, sizeof m, fmt, arg ? arg : "");
  cJSON* o = cJSON_CreateObject();
  cJSON* e = cJSON_AddObjectToObject(o, "error");
  cJSON_AddStringToObject(e, "message", m);
  cJSON_AddStringToObject(e, "type", "invalid_request_error");
  return o;
}

/* the server's baseline state for a handle: domain, every section, no filters / redaction / overrides, then the
 * ARCTAG_CONFIG file */
static int apply_baseline(arctag_t* h) {
  int rc = arctag_set_option(h, "domain", G.domain) | arctag_set_option(h, "asr", "unknown") |
           arctag_set_option(h, "emit", "all") | arctag_set_option(h, "tags", "") | arctag_set_option(h, "redact", "off") |
           arctag_set_option(h, "thr_reset", "") | arctag_set_option(h, "fine_auto", "on") |
           arctag_set_option(h, "guard", "auto") | arctag_set_option(h, "command_context", "off");
  if (G.config_json) rc |= arctag_apply_config(h, G.config_json);
  arctag_reset(h);
  return rc ? -1 : 0;
}

static arctag_t* new_handle(char* err, size_t el) {
  arctag_t* h = arctag_create_ex(G.model_dir, &G.opt, err, el);
  if (!h) return NULL;
  if (apply_baseline(h)) {
    snprintf(err, el, "config: %s", arctag_last_error(h));
    arctag_destroy(h);
    return NULL;
  }
  return h;
}

int tagsrv_init(const char* model_dir, char* err, size_t el) {
  snprintf(G.model_dir, sizeof G.model_dir, "%s", model_dir);
  snprintf(G.model_id, sizeof G.model_id, "%s", env_or("ARCTAG_MODEL_ID", "arc-tagger"));
  snprintf(G.domain, sizeof G.domain, "%s", env_or("ARCTAG_DOMAIN", "auto"));
  arctag_options_init(&G.opt);
  G.opt.precision = !strcmp(env_or("ARCTAG_PRECISION", "fp32"), "int8") ? ARCTAG_INT8 : ARCTAG_FP32;
  G.opt.backend = ARCTAG_BACKEND_NATIVE;
  G.opt.threads = 1;
  const char* cfg = getenv("ARCTAG_CONFIG");
  if (cfg && *cfg) {
    G.config_json = read_file(cfg);
    if (!G.config_json) { snprintf(err, el, "cannot read ARCTAG_CONFIG %s", cfg); return -1; }
    /* the file may carry load-time keys; those are applied at load by arctag_apply_config too */
  }
  G.n = atoi(env_or("ARCTAG_WORKERS", "2"));
  if (G.n < 1) G.n = 1;
  if (G.n > 64) G.n = 64;
  G.max_streams = atoi(env_or("ARCTAG_MAX_STREAMS", "16"));
  G.queue_timeout_s = atoi(env_or("ARC_QUEUE_TIMEOUT_S", "60"));
  G.h = calloc((size_t)G.n, sizeof *G.h);
  G.busy = calloc((size_t)G.n, sizeof *G.busy);
  G.dirty = calloc((size_t)G.n, sizeof *G.dirty);
  for (int i = 0; i < G.n; ++i) {
    G.h[i] = new_handle(err, el);
    if (!G.h[i]) return -1;
  }
  G.ready = 1;
  return 0;
}

void tagsrv_shutdown(void) {
  G.ready = 0;
  for (int i = 0; i < G.n; ++i) if (G.h[i]) arctag_destroy(G.h[i]);
  free(G.h);
  free(G.busy);
  free(G.dirty);
  free(G.config_json);
  G.h = NULL;
  G.n = 0;
}

int tagsrv_ready(void) { return G.ready; }
const char* tagsrv_model_id(void) { return G.model_id; }
const char* tagsrv_model_dir(void) { return G.model_dir; }

static int acquire(void) {
  struct timespec dl;
  clock_gettime(CLOCK_REALTIME, &dl);
  dl.tv_sec += G.queue_timeout_s > 0 ? G.queue_timeout_s : 60;
  pthread_mutex_lock(&G.mu);
  for (;;) {
    for (int i = 0; i < G.n; ++i)
      if (!G.busy[i]) {
        G.busy[i] = 1;
        pthread_mutex_unlock(&G.mu);
        return i;
      }
    if (pthread_cond_timedwait(&G.cv, &G.mu, &dl) == ETIMEDOUT) {
      pthread_mutex_unlock(&G.mu);
      return -1;
    }
  }
}

static void release(int i) {
  if (G.dirty[i]) {
    apply_baseline(G.h[i]);
    G.dirty[i] = 0;
  } else arctag_reset(G.h[i]);
  pthread_mutex_lock(&G.mu);
  G.busy[i] = 0;
  pthread_cond_signal(&G.cv);
  pthread_mutex_unlock(&G.mu);
}

cJSON* tagsrv_info(void) {
  int i = acquire();
  if (i < 0) return NULL;
  char* s = arctag_info(G.h[i]);
  release(i);
  cJSON* o = s ? cJSON_Parse(s) : NULL;
  arctag_free_str(s);
  return o;
}

/* per-request option keys (a subset of the config keys: nothing that reloads the model) */
static const char* OPT_KEYS[] = {"domain", "asr", "emit", "tags", "redact", "thresholds", "fine_auto", "guard", "command_context", NULL};
static const char* LOAD_KEYS[] = {"precision", "backend", "threads", "max_ctx", "stream_ctx", "stream_state", NULL};

/* the options of a request as one config object (NULL: none); a string emit / tags is split on commas */
static cJSON* request_config(const cJSON* req, char* err, size_t el) {
  cJSON* cfg = NULL;
  const cJSON* inner = cJSON_GetObjectItemCaseSensitive(req, "config");
  for (int pass = 0; pass < 2; ++pass) {
    const cJSON* src = pass ? req : inner;
    if (!cJSON_IsObject(src)) continue;
    for (const cJSON* x = src->child; x; x = x->next) {
      for (int k = 0; LOAD_KEYS[k]; ++k)
        if (!strcmp(x->string, LOAD_KEYS[k])) {
          snprintf(err, el, "'%s' is fixed when the server starts (env ARCTAG_*), not per request", x->string);
          cJSON_Delete(cfg);
          return (cJSON*)-1;
        }
      int ok = 0;
      for (int k = 0; OPT_KEYS[k]; ++k) ok |= !strcmp(x->string, OPT_KEYS[k]);
      if (!ok) {
        if (pass == 0 && x->string[0] != '_') {
          snprintf(err, el, "config: unknown key '%s'", x->string);
          cJSON_Delete(cfg);
          return (cJSON*)-1;
        }
        continue;
      }
      if (!cfg) cfg = cJSON_CreateObject();
      cJSON_DeleteItemFromObjectCaseSensitive(cfg, x->string);
      if ((!strcmp(x->string, "emit") || !strcmp(x->string, "tags")) && cJSON_IsString(x)) {
        cJSON* arr = cJSON_CreateArray();
        char* s = strdup(x->valuestring);
        for (char* t = strtok(s, ", "); t; t = strtok(NULL, ", ")) cJSON_AddItemToArray(arr, cJSON_CreateString(t));
        free(s);
        cJSON_AddItemToObject(cfg, x->string, arr);
      } else cJSON_AddItemToObject(cfg, x->string, cJSON_Duplicate(x, 1));
    }
  }
  return cfg;
}

static int apply_cfg(arctag_t* h, const cJSON* cfg, char* err, size_t el) {
  if (!cfg) return 0;
  char* s = cJSON_PrintUnformatted(cfg);
  int rc = arctag_apply_config(h, s);
  free(s);
  if (rc) snprintf(err, el, "%s", arctag_last_error(h));
  return rc;
}

static void spread_words(cJSON* out, const char* text, double t0, double t1, const cJSON* spk) {
  char* s = strdup(text ? text : "");
  int n = 0;
  for (char* p = s; *p;) {
    while (*p && isspace((unsigned char)*p)) ++p;
    if (!*p) break;
    ++n;
    while (*p && !isspace((unsigned char)*p)) ++p;
  }
  int k = 0;
  char* save = NULL;
  for (char* t = strtok_r(s, " \t\r\n", &save); t; t = strtok_r(NULL, " \t\r\n", &save), ++k) {
    cJSON* w = cJSON_CreateObject();
    cJSON_AddStringToObject(w, "w", t);
    if (t1 > t0 && n) {
      cJSON_AddNumberToObject(w, "t0", t0 + (t1 - t0) * k / n);
      cJSON_AddNumberToObject(w, "t1", t0 + (t1 - t0) * (k + 1) / n);
    }
    if (spk) cJSON_AddItemToObject(w, "spk", cJSON_Duplicate(spk, 1));
    cJSON_AddItemToArray(out, w);
  }
  free(s);
}

cJSON* tagsrv_segments_to_words(const cJSON* segs) {
  if (!cJSON_IsArray(segs)) return NULL;
  cJSON* out = cJSON_CreateArray();
  const cJSON* seg;
  cJSON_ArrayForEach(seg, segs) {
    int before = cJSON_GetArraySize(out);
    const cJSON* spk = cJSON_GetObjectItemCaseSensitive(seg, "speaker");
    if (!spk) spk = cJSON_GetObjectItemCaseSensitive(seg, "spk");
    if (spk && !cJSON_IsString(spk) && !cJSON_IsNumber(spk)) spk = NULL;
    const cJSON* words = cJSON_GetObjectItemCaseSensitive(seg, "words");
    if (cJSON_IsArray(words) && cJSON_GetArraySize(words) > 0) {
      const cJSON* w;
      cJSON_ArrayForEach(w, words) {
        cJSON* o = cJSON_CreateObject();
        const cJSON* t = cJSON_GetObjectItemCaseSensitive(w, "word");
        if (!cJSON_IsString(t)) t = cJSON_GetObjectItemCaseSensitive(w, "text");
        if (!cJSON_IsString(t)) t = cJSON_GetObjectItemCaseSensitive(w, "w");
        if (cJSON_IsString(w)) t = w;
        if (!cJSON_IsString(t)) { cJSON_Delete(o); continue; }
        cJSON_AddStringToObject(o, "w", t->valuestring);
        static const char* K[][2] = {{"start", "t0"}, {"end", "t1"}, {"t0", "t0"}, {"t1", "t1"}, {"conf", "conf"}, {"confidence", "conf"}};
        for (int k = 0; k < 6; ++k) {
          const cJSON* v = cJSON_IsObject(w) ? cJSON_GetObjectItemCaseSensitive(w, K[k][0]) : NULL;
          if (cJSON_IsNumber(v) && !cJSON_GetObjectItemCaseSensitive(o, K[k][1]) && (strcmp(K[k][1], "conf") || v->valuedouble >= 0))
            cJSON_AddNumberToObject(o, K[k][1], v->valuedouble);
        }
        const cJSON* ws = cJSON_IsObject(w) ? cJSON_GetObjectItemCaseSensitive(w, "speaker") : NULL;
        if (ws && (cJSON_IsString(ws) || cJSON_IsNumber(ws))) cJSON_AddItemToObject(o, "spk", cJSON_Duplicate(ws, 1));
        else if (spk) cJSON_AddItemToObject(o, "spk", cJSON_Duplicate(spk, 1));
        cJSON_AddItemToArray(out, o);
      }
    } else {
      const cJSON* t = cJSON_GetObjectItemCaseSensitive(seg, "text");
      const cJSON* s0 = cJSON_GetObjectItemCaseSensitive(seg, "start");
      const cJSON* s1 = cJSON_GetObjectItemCaseSensitive(seg, "end");
      spread_words(out, cJSON_IsString(t) ? t->valuestring : "", cJSON_IsNumber(s0) ? s0->valuedouble : 0,
                   cJSON_IsNumber(s1) ? s1->valuedouble : 0, spk);
    }
    int after = cJSON_GetArraySize(out);
    if (after > before) cJSON_AddBoolToObject(cJSON_GetArrayItem(out, after - 1), "eou", 1);
  }
  if (cJSON_GetArraySize(out) == 0) { cJSON_Delete(out); return NULL; }
  return out;
}

/* one non-batch request on handle i */
static cJSON* run_one(int i, const cJSON* req, int* status) {
  char err[1024] = "";
  cJSON* cfg = request_config(req, err, sizeof err);
  if (cfg == (cJSON*)-1) { *status = 400; return err_obj("%s", err); }
  if (cfg) {
    G.dirty[i] = 1;
    if (apply_cfg(G.h[i], cfg, err, sizeof err)) {
      cJSON_Delete(cfg);
      *status = 400;
      return err_obj("%s", err);
    }
    cJSON_Delete(cfg);
  }
  const cJSON* text = cJSON_GetObjectItemCaseSensitive(req, "text");
  const cJSON* words = cJSON_GetObjectItemCaseSensitive(req, "words");
  const cJSON* segs = cJSON_GetObjectItemCaseSensitive(req, "segments");
  char* out = NULL;
  if (cJSON_IsArray(words)) {
    char* w = cJSON_PrintUnformatted(words);
    out = arctag_tag_words(G.h[i], w);
    free(w);
  } else if (cJSON_IsArray(segs)) {
    cJSON* wa = tagsrv_segments_to_words(segs);
    if (!wa) { *status = 400; return err_obj("segments hold no words", NULL); }
    char* w = cJSON_PrintUnformatted(wa);
    cJSON_Delete(wa);
    out = arctag_tag_words(G.h[i], w);
    free(w);
  } else if (cJSON_IsString(text)) {
    out = arctag_tag_text(G.h[i], text->valuestring);
  } else {
    *status = 400;
    return err_obj("give one of: text (string), words (array), segments (array), inputs (batch array)", NULL);
  }
  if (!out) { *status = 400; return err_obj("%s", arctag_last_error(G.h[i])); }
  cJSON* o = cJSON_Parse(out);
  arctag_free_str(out);
  if (!o) { *status = 500; return err_obj("internal: unparsable tagger output", NULL); }
  *status = 200;
  return o;
}

cJSON* tagsrv_process(const cJSON* req, int* status) {
  if (!cJSON_IsObject(req)) { *status = 400; return err_obj("the body must be a JSON object", NULL); }
  const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(req, "inputs");
  if (!inputs) inputs = cJSON_GetObjectItemCaseSensitive(req, "instances");   /* SageMaker / Vertex style */
  int i = acquire();
  if (i < 0) { *status = 503; return err_obj("all tagger workers busy", NULL); }
  cJSON* res;
  if (cJSON_IsArray(inputs)) {
    res = cJSON_CreateObject();
    cJSON_AddStringToObject(res, "model", G.model_id);
    cJSON* arr = cJSON_AddArrayToObject(res, "results");
    const cJSON* it;
    cJSON_ArrayForEach(it, inputs) {
      int st = 0;
      cJSON* one;
      if (cJSON_IsString(it) || cJSON_IsObject(it)) {
        /* batch-level options apply to every item that does not set its own */
        cJSON* merged;
        if (cJSON_IsString(it)) {
          merged = cJSON_CreateObject();
          cJSON_AddStringToObject(merged, "text", it->valuestring);
        } else merged = cJSON_Duplicate(it, 1);
        for (int k = 0; OPT_KEYS[k]; ++k) {
          const cJSON* v = cJSON_GetObjectItemCaseSensitive(req, OPT_KEYS[k]);
          if (v && !cJSON_GetObjectItemCaseSensitive(merged, OPT_KEYS[k])) cJSON_AddItemToObject(merged, OPT_KEYS[k], cJSON_Duplicate(v, 1));
        }
        one = run_one(i, merged, &st);
        cJSON_Delete(merged);
      } else {
        one = err_obj("each input is a string or an object", NULL);
      }
      if (G.dirty[i]) { apply_baseline(G.h[i]); G.dirty[i] = 0; }
      arctag_reset(G.h[i]);
      cJSON_AddItemToArray(arr, one);
    }
    *status = 200;
  } else {
    res = run_one(i, req, status);
    if (*status == 200) cJSON_AddStringToObject(res, "model", G.model_id);
  }
  release(i);
  return res;
}

cJSON* tagsrv_tag_words(const char* words_json, const cJSON* opts, int* status) {
  int i = acquire();
  if (i < 0) { *status = 503; return err_obj("all tagger workers busy", NULL); }
  char err[1024] = "";
  cJSON* res = NULL;
  if (opts) {
    cJSON* cfg = request_config(opts, err, sizeof err);
    if (cfg == (cJSON*)-1) { *status = 400; release(i); return err_obj("%s", err); }
    if (cfg) {
      G.dirty[i] = 1;
      int rc = apply_cfg(G.h[i], cfg, err, sizeof err);
      cJSON_Delete(cfg);
      if (rc) { *status = 400; release(i); return err_obj("%s", err); }
    }
  }
  char* out = arctag_tag_words(G.h[i], words_json);
  if (!out) { *status = 400; res = err_obj("%s", arctag_last_error(G.h[i])); }
  else {
    res = cJSON_Parse(out);
    arctag_free_str(out);
    *status = res ? 200 : 500;
    if (!res) res = err_obj("internal: unparsable tagger output", NULL);
  }
  release(i);
  return res;
}

/* ------------------------------------------------------------------ streams */
struct tagsrv_stream {
  arctag_t* h;
};

tagsrv_stream* tagsrv_stream_open(const cJSON* opts, char* err, size_t el) {
  if (atomic_fetch_add(&G.streams, 1) >= G.max_streams) {
    atomic_fetch_sub(&G.streams, 1);
    snprintf(err, el, "too many realtime sessions (ARCTAG_MAX_STREAMS=%d)", G.max_streams);
    return NULL;
  }
  arctag_t* h = new_handle(err, el);
  if (!h) { atomic_fetch_sub(&G.streams, 1); return NULL; }
  tagsrv_stream* s = calloc(1, sizeof *s);
  s->h = h;
  if (opts && tagsrv_stream_apply(s, opts, err, el)) { tagsrv_stream_close(s); return NULL; }
  return s;
}

int tagsrv_stream_apply(tagsrv_stream* s, const cJSON* opts, char* err, size_t el) {
  cJSON* cfg = request_config(opts, err, el);
  if (cfg == (cJSON*)-1) return -1;
  int rc = apply_cfg(s->h, cfg, err, el);
  cJSON_Delete(cfg);
  return rc;
}

cJSON* tagsrv_stream_push(tagsrv_stream* s, const char* push_json) {
  if (push_json && arctag_push_words(s->h, push_json)) return err_obj("%s", arctag_last_error(s->h));
  char* p = arctag_poll(s->h);
  if (!p) return err_obj("%s", arctag_last_error(s->h));
  cJSON* o = cJSON_Parse(p);
  arctag_free_str(p);
  return o ? o : err_obj("internal: unparsable poll output", NULL);
}

void tagsrv_stream_reset(tagsrv_stream* s) { arctag_reset(s->h); }

void tagsrv_stream_close(tagsrv_stream* s) {
  if (!s) return;
  arctag_destroy(s->h);
  free(s);
  atomic_fetch_sub(&G.streams, 1);
}

/* ------------------------------------------------------------------ WS /v1/realtime */
static void ws_json(int fd, cJSON* o) {
  char* s = cJSON_PrintUnformatted(o);
  ws_send_text(fd, s);
  free(s);
  cJSON_Delete(o);
}

static void ws_error(int fd, const char* msg) {
  cJSON* o = cJSON_CreateObject();
  cJSON_AddStringToObject(o, "type", "error");
  cJSON_AddStringToObject(o, "message", msg);
  ws_json(fd, o);
}

static void send_update(int fd, cJSON* poll) {
  if (cJSON_GetObjectItemCaseSensitive(poll, "error")) {
    const cJSON* e = cJSON_GetObjectItemCaseSensitive(cJSON_GetObjectItemCaseSensitive(poll, "error"), "message");
    ws_error(fd, cJSON_IsString(e) ? e->valuestring : "error");
    cJSON_Delete(poll);
    return;
  }
  cJSON_AddStringToObject(poll, "type", "tags.update");
  ws_json(fd, poll);
}

void tagsrv_ws_session(http_req* r) {
  char err[512] = "";
  /* query options (domain, asr, emit, redact) */
  cJSON* q = cJSON_CreateObject();
  const char* qk[] = {"domain", "asr", "emit", "tags", "redact", NULL};
  for (int k = 0; qk[k]; ++k) {
    char v[512];
    if (http_query(r, qk[k], v, sizeof v)) cJSON_AddStringToObject(q, qk[k], v);
  }
  tagsrv_stream* s = tagsrv_stream_open(q, err, sizeof err);
  cJSON_Delete(q);
  if (!s) {
    http_respond_error(r, *err && strstr(err, "too many") ? 503 : 400, "invalid_request_error", "%s", err);
    return;
  }
  if (ws_accept(r)) { tagsrv_stream_close(s); return; }
  int fd = r->fd;
  {
    cJSON* o = cJSON_CreateObject();
    cJSON_AddStringToObject(o, "type", "session.created");
    cJSON_AddStringToObject(o, "model", G.model_id);
    char* info = arctag_info(s->h);
    cJSON* io = info ? cJSON_Parse(info) : NULL;
    arctag_free_str(info);
    if (io) cJSON_AddItemToObject(o, "session", io);
    ws_json(fd, o);
  }
  for (;;) {
    unsigned char* data;
    size_t len;
    int op = ws_recv(fd, &data, &len, (size_t)8 << 20);
    if (op == WS_CLOSE || op < 0) break;
    if (op == WS_BINARY) { free(data); ws_error(fd, "the tagger takes text frames (JSON)"); continue; }
    cJSON* m = cJSON_Parse((const char*)data);
    free(data);
    if (!m) { ws_error(fd, "invalid JSON"); continue; }
    if (cJSON_IsArray(m)) {   /* a bare words array = append */
      char* w = cJSON_PrintUnformatted(m);
      send_update(fd, tagsrv_stream_push(s, w));
      free(w);
      cJSON_Delete(m);
      continue;
    }
    const cJSON* ty = cJSON_GetObjectItemCaseSensitive(m, "type");
    const char* t = cJSON_IsString(ty) ? ty->valuestring : "";
    const cJSON* ep = cJSON_GetObjectItemCaseSensitive(m, "endpoint");
    int endpoint = cJSON_IsTrue(ep);
    if (!strcmp(t, "session.update")) {
      const cJSON* ses = cJSON_GetObjectItemCaseSensitive(m, "session");
      if (tagsrv_stream_apply(s, ses ? ses : m, err, sizeof err)) ws_error(fd, err);
      else {
        cJSON* o = cJSON_CreateObject();
        cJSON_AddStringToObject(o, "type", "session.updated");
        char* info = arctag_info(s->h);
        cJSON* io = info ? cJSON_Parse(info) : NULL;
        arctag_free_str(info);
        if (io) cJSON_AddItemToObject(o, "session", io);
        ws_json(fd, o);
      }
    } else if (!strcmp(t, "words") || (!*t && cJSON_GetObjectItemCaseSensitive(m, "words"))) {
      cJSON* p = cJSON_CreateObject();
      const cJSON* w = cJSON_GetObjectItemCaseSensitive(m, "words");
      const cJSON* from = cJSON_GetObjectItemCaseSensitive(m, "from");
      cJSON_AddItemToObject(p, "words", w ? cJSON_Duplicate(w, 1) : cJSON_CreateArray());
      if (cJSON_IsNumber(from)) cJSON_AddNumberToObject(p, "from", from->valuedouble);
      if (endpoint) cJSON_AddBoolToObject(p, "endpoint", 1);
      char* js = cJSON_PrintUnformatted(p);
      cJSON_Delete(p);
      send_update(fd, tagsrv_stream_push(s, js));
      free(js);
    } else if (!strcmp(t, "text") || (!*t && cJSON_GetObjectItemCaseSensitive(m, "text"))) {
      const cJSON* tx = cJSON_GetObjectItemCaseSensitive(m, "text");
      cJSON* arr = cJSON_CreateArray();
      spread_words(arr, cJSON_IsString(tx) ? tx->valuestring : "", 0, 0, cJSON_GetObjectItemCaseSensitive(m, "speaker"));
      cJSON* p = cJSON_CreateObject();
      cJSON_AddItemToObject(p, "words", arr);
      if (endpoint) cJSON_AddBoolToObject(p, "endpoint", 1);
      char* js = cJSON_PrintUnformatted(p);
      cJSON_Delete(p);
      send_update(fd, tagsrv_stream_push(s, js));
      free(js);
    } else if (!strcmp(t, "endpoint")) {
      send_update(fd, tagsrv_stream_push(s, "{\"endpoint\":true}"));
    } else if (!strcmp(t, "poll")) {
      send_update(fd, tagsrv_stream_push(s, NULL));
    } else if (!strcmp(t, "reset")) {
      tagsrv_stream_reset(s);
      cJSON* o = cJSON_CreateObject();
      cJSON_AddStringToObject(o, "type", "reset");
      ws_json(fd, o);
    } else if (!strcmp(t, "session.close")) {
      cJSON_Delete(m);
      ws_close(fd, 1000, "bye");
      break;
    } else {
      ws_error(fd, "unknown type (session.update | words | text | endpoint | poll | reset | session.close)");
    }
    cJSON_Delete(m);
  }
  tagsrv_stream_close(s);
}
