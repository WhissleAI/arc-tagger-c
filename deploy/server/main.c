/* arctag-serve: Arc Tagger as an HTTP + WebSocket service (native C backend, no Python, no network at run time).
 *
 *   arctag-serve [serve] [--port N] [--host ADDR] [--model DIR]
 *
 * Routes: POST /v1/tags, WS /v1/realtime, GET /health, GET /v1/health/live, GET /v1/health/ready, GET /v1/models,
 * and SageMaker's GET /ping, POST /invocations, GET /execution-parameters. See deploy/openapi.yaml.
 * Environment: PORT (8080), HOST (0.0.0.0), ARCTAG_MODEL_DIR (else the first of /opt/ml/model, /repository,
 * /opt/arc/model holding arctag_meta.json, also in an onnx/ or arc-tagger/ subfolder), ARC_MAX_BODY_MB (100),
 * ARC_MAX_CONNECTIONS (64), ARC_ACCESS_LOG (1), and the ARCTAG_* settings of tagsrv.h.
 * Part of arc-tagger-c (Whissle Inc.), CC BY-NC 4.0 with the Whissle terms; see LICENSE. */
#define _GNU_SOURCE   /* strcasestr, M_PI, strdup, strtok_r, clock_gettime on glibc with -std=c11 */
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "arctag.h"
#include "cJSON.h"
#include "http.h"
#include "tagsrv.h"

#ifndef ARC_DEPLOY_VERSION
#define ARC_DEPLOY_VERSION "0.1.0"
#endif

static time_t g_started;

static int has_meta(const char* dir) {
  char p[1200];
  struct stat st;
  snprintf(p, sizeof p, "%s/arctag_meta.json", dir);
  return stat(p, &st) == 0;
}

/* the model folder: $ARCTAG_MODEL_DIR, or a platform's mount point (SageMaker /opt/ml/model, Hugging Face
 * /repository, the image's own /opt/arc/model), or a subfolder of one */
static int find_model_dir(const char* given, char* out, size_t n) {
  const char* roots[] = {given, getenv("ARCTAG_MODEL_DIR"), "/opt/ml/model", "/repository", "/opt/arc/model", "model", NULL};
  const char* subs[] = {"", "/onnx", "/arc-tagger", "/arc-tagger/onnx", "/tagger", NULL};
  for (int i = 0; i < 6; ++i) {
    if (!roots[i] || !*roots[i]) continue;
    for (int k = 0; subs[k]; ++k) {
      snprintf(out, n, "%s%s", roots[i], subs[k]);
      if (has_meta(out)) return 0;
    }
    if (i < 2) { snprintf(out, n, "%s", roots[i]); return -1; }   /* given explicitly but not a model folder */
  }
  return -1;
}

static void respond_obj(http_req* r, int status, cJSON* o) {
  char* s = cJSON_PrintUnformatted(o);
  http_respond_json(r, status, s);
  free(s);
  cJSON_Delete(o);
}

static void models(http_req* r, const char* only) {
  cJSON* o = cJSON_CreateObject();
  cJSON* m = cJSON_CreateObject();
  cJSON_AddStringToObject(m, "id", tagsrv_model_id());
  cJSON_AddStringToObject(m, "object", "model");
  cJSON_AddNumberToObject(m, "created", (double)g_started);
  cJSON_AddStringToObject(m, "owned_by", "whissle");
  cJSON* caps = cJSON_AddArrayToObject(m, "capabilities");
  cJSON_AddItemToArray(caps, cJSON_CreateString("tags"));
  cJSON_AddItemToArray(caps, cJSON_CreateString("realtime"));
  cJSON* info = tagsrv_info();
  if (info) cJSON_AddItemToObject(m, "arctag", info);
  cJSON_AddStringToObject(m, "server_version", ARC_DEPLOY_VERSION);
  if (only) {
    if (strcmp(only, tagsrv_model_id())) {
      cJSON_Delete(o);
      cJSON_Delete(m);
      http_respond_error(r, 404, "invalid_request_error", "model '%s' not found", only);
      return;
    }
    cJSON_Delete(o);
    respond_obj(r, 200, m);
    return;
  }
  cJSON_AddStringToObject(o, "object", "list");
  cJSON* d = cJSON_AddArrayToObject(o, "data");
  cJSON_AddItemToArray(d, m);
  respond_obj(r, 200, o);
}

/* POST /v1/tags and JSON /invocations */
static void tags(http_req* r) {
  cJSON* req = NULL;
  if (http_ctype_is(r, "application/json") || (!http_header(r, "Content-Type") && r->body_len && r->body[0] == '{')) {
    req = cJSON_ParseWithLength(r->body, r->body_len);
    if (!req) { http_respond_error(r, 400, "invalid_request_error", "invalid JSON body"); return; }
  } else if (http_ctype_is(r, "text/plain") || !http_header(r, "Content-Type")) {
    req = cJSON_CreateObject();
    cJSON_AddStringToObject(req, "text", r->body);
    char v[64];
    if (http_query(r, "domain", v, sizeof v)) cJSON_AddStringToObject(req, "domain", v);
  } else {
    http_respond_error(r, 415, "invalid_request_error", "send application/json or text/plain (the tagger takes text, not audio)");
    return;
  }
  if (cJSON_IsArray(req)) {   /* a bare array: a batch */
    cJSON* w = cJSON_CreateObject();
    cJSON_AddItemToObject(w, "inputs", req);
    req = w;
  }
  int st = 0;
  cJSON* out = tagsrv_process(req, &st);
  cJSON_Delete(req);
  respond_obj(r, st, out);
}

/* application/jsonlines: one request per line (SageMaker batch transform with SplitType=Line), one result per line */
static void tags_jsonl(http_req* r) {
  size_t cap = 4096, len = 0;
  char* out = malloc(cap);
  char* save = NULL;
  for (char* line = strtok_r(r->body, "\n", &save); line; line = strtok_r(NULL, "\n", &save)) {
    while (*line == ' ' || *line == '\r') ++line;
    if (!*line) continue;
    cJSON* req = cJSON_Parse(line);
    int st = 0;
    cJSON* res;
    if (!req) {
      res = cJSON_CreateObject();
      cJSON_AddStringToObject(cJSON_AddObjectToObject(res, "error"), "message", "invalid JSON line");
    } else res = tagsrv_process(req, &st);
    cJSON_Delete(req);
    char* s = cJSON_PrintUnformatted(res);
    cJSON_Delete(res);
    size_t sl = strlen(s);
    while (len + sl + 2 > cap) cap *= 2;
    out = realloc(out, cap);
    memcpy(out + len, s, sl);
    len += sl;
    out[len++] = '\n';
    free(s);
  }
  http_respond(r, 200, "application/jsonlines", out, len);
  free(out);
}

/* Snowflake service functions (SPCS): {"data": [[row, text, domain?, options_json?], ...]} -> {"data": [[row, result], ...]}
 *   CREATE FUNCTION arc_tags(text VARCHAR) RETURNS VARIANT SERVICE=arc_tagger ENDPOINT=api AS '/v1/snowflake/tags'; */
static void snowflake_tags(http_req* r) {
  cJSON* j = cJSON_ParseWithLength(r->body, r->body_len);
  const cJSON* rows = j ? cJSON_GetObjectItemCaseSensitive(j, "data") : NULL;
  if (!cJSON_IsArray(rows)) {
    cJSON_Delete(j);
    http_respond_error(r, 400, "invalid_request_error", "expected {\"data\": [[row, text, ...], ...]}");
    return;
  }
  cJSON* out = cJSON_CreateObject();
  cJSON* data = cJSON_AddArrayToObject(out, "data");
  const cJSON* row;
  cJSON_ArrayForEach(row, rows) {
    const cJSON* idx = cJSON_GetArrayItem(row, 0);
    const cJSON* text = cJSON_GetArrayItem(row, 1);
    const cJSON* dom = cJSON_GetArrayItem(row, 2);
    const cJSON* opt = cJSON_GetArrayItem(row, 3);
    cJSON* req = NULL;
    if (cJSON_IsString(opt) && opt->valuestring[0] == '{') req = cJSON_Parse(opt->valuestring);
    else if (cJSON_IsObject(opt)) req = cJSON_Duplicate(opt, 1);
    if (!req) req = cJSON_CreateObject();
    cJSON_DeleteItemFromObjectCaseSensitive(req, "text");
    cJSON_AddStringToObject(req, "text", cJSON_IsString(text) ? text->valuestring : "");
    if (cJSON_IsString(dom) && *dom->valuestring) {
      cJSON_DeleteItemFromObjectCaseSensitive(req, "domain");
      cJSON_AddStringToObject(req, "domain", dom->valuestring);
    }
    int st = 0;
    cJSON* res = tagsrv_process(req, &st);
    cJSON_Delete(req);
    cJSON* pair = cJSON_CreateArray();
    cJSON_AddItemToArray(pair, idx ? cJSON_Duplicate(idx, 1) : cJSON_CreateNull());
    cJSON_AddItemToArray(pair, res);
    cJSON_AddItemToArray(data, pair);
  }
  cJSON_Delete(j);
  respond_obj(r, 200, out);
}

static void handler(http_req* r, void* user) {
  (void)user;
  const char* p = r->path;
  int get = !strcmp(r->method, "GET") || !strcmp(r->method, "HEAD"), post = !strcmp(r->method, "POST");
  if (get && (!strcmp(p, "/health") || !strcmp(p, "/v1/health/live") || !strcmp(p, "/healthz"))) {
    http_respond_json(r, 200, "{\"status\":\"ok\"}");
  } else if (get && (!strcmp(p, "/v1/health/ready") || !strcmp(p, "/ping") || !strcmp(p, "/readyz"))) {
    if (tagsrv_ready()) http_respond_json(r, 200, "{\"status\":\"ready\"}");
    else http_respond_json(r, 503, "{\"status\":\"loading\"}");
  } else if (get && !strcmp(p, "/v1/models")) {
    models(r, NULL);
  } else if (get && !strncmp(p, "/v1/models/", 11)) {
    models(r, p + 11);
  } else if (!strcmp(p, "/v1/realtime")) {
    if (!http_is_ws_upgrade(r)) http_respond_error(r, 400, "invalid_request_error", "WebSocket upgrade required");
    else tagsrv_ws_session(r);
  } else if (!strcmp(p, "/v1/tags") || !strcmp(p, "/invocations")) {
    if (!post) { http_respond_error(r, 405, "invalid_request_error", "use POST"); return; }
    if (http_ctype_is(r, "application/jsonlines") || http_ctype_is(r, "application/x-jsonlines") || http_ctype_is(r, "application/jsonl"))
      tags_jsonl(r);
    else tags(r);
  } else if (!strcmp(p, "/v1/snowflake/tags")) {
    if (!post) http_respond_error(r, 405, "invalid_request_error", "use POST");
    else snowflake_tags(r);
  } else if (get && !strcmp(p, "/execution-parameters")) {
    char b[256];
    snprintf(b, sizeof b, "{\"MaxConcurrentTransforms\":%d,\"BatchStrategy\":\"MULTI_RECORD\",\"MaxPayloadInMB\":6}",
             atoi(getenv("ARCTAG_WORKERS") ? getenv("ARCTAG_WORKERS") : "2") > 0 ? atoi(getenv("ARCTAG_WORKERS") ? getenv("ARCTAG_WORKERS") : "2") : 1);
    http_respond_json(r, 200, b);
  } else if (get && !strcmp(p, "/")) {
    char b[512];
    snprintf(b, sizeof b, "{\"service\":\"arc-tagger\",\"model\":\"%s\",\"version\":\"%s\",\"arctag\":\"%s\",\"openapi\":\"/openapi.yaml\"}",
             tagsrv_model_id(), ARC_DEPLOY_VERSION, arctag_version());
    http_respond_json(r, 200, b);
  } else if (get && !strcmp(p, "/openapi.yaml")) {
    const char* f = getenv("ARC_OPENAPI") ? getenv("ARC_OPENAPI") : "/opt/arc/openapi.yaml";
    FILE* fp = fopen(f, "rb");
    if (!fp) { http_respond_error(r, 404, "invalid_request_error", "openapi.yaml not bundled"); return; }
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char* s = malloc((size_t)n + 1);
    size_t got = fread(s, 1, (size_t)n, fp);
    fclose(fp);
    http_respond(r, 200, "application/yaml", s, got);
    free(s);
  } else {
    http_respond_error(r, 404, "invalid_request_error", "no route %s %s", r->method, p);
  }
}

static void on_term(int sig) {
  (void)sig;
  http_server_stop();
  _exit(0);
}

int main(int argc, char** argv) {
  const char* port_s = getenv("PORT") ? getenv("PORT") : (getenv("SAGEMAKER_BIND_TO_PORT") ? getenv("SAGEMAKER_BIND_TO_PORT") : "8080");
  const char* host = getenv("HOST") ? getenv("HOST") : "0.0.0.0";
  const char* model = NULL;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "serve")) continue;   /* SageMaker runs the image as `docker run IMAGE serve` */
    else if (!strcmp(argv[i], "--port") && i + 1 < argc) port_s = argv[++i];
    else if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
    else if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
    else if (!strcmp(argv[i], "--healthcheck")) return http_healthcheck(atoi(port_s), "/v1/health/ready");
    else if (!strcmp(argv[i], "--version")) { printf("arctag-serve %s (arctag %s)\n", ARC_DEPLOY_VERSION, arctag_version()); return 0; }
    else {
      fprintf(stderr, "usage: %s [serve] [--port N] [--host ADDR] [--model DIR]\n", argv[0]);
      return 2;
    }
  }
  char dir[1200], err[1024];
  if (find_model_dir(model, dir, sizeof dir)) {
    fprintf(stderr, "arctag-serve: no model folder (arctag_meta.json, tokenizer.json, arctag[_int8].onnx) at %s; set ARCTAG_MODEL_DIR "
                    "or mount one at /opt/ml/model or /repository\n", model ? model : (getenv("ARCTAG_MODEL_DIR") ? dir : "the default paths"));
    return 1;
  }
  g_started = time(NULL);
  double t0 = http_now_ms();
  if (tagsrv_init(dir, err, sizeof err)) {
    fprintf(stderr, "arctag-serve: %s: %s\n", dir, err);
    return 1;
  }
  signal(SIGTERM, on_term);
  signal(SIGINT, on_term);
  http_server_config c = {0};
  c.host = host;
  c.port = atoi(port_s);
  c.max_connections = getenv("ARC_MAX_CONNECTIONS") ? atoi(getenv("ARC_MAX_CONNECTIONS")) : 64;
  c.max_body = (size_t)(getenv("ARC_MAX_BODY_MB") ? atoi(getenv("ARC_MAX_BODY_MB")) : 100) << 20;
  c.access_log = getenv("ARC_ACCESS_LOG") ? atoi(getenv("ARC_ACCESS_LOG")) : 1;
  c.handler = handler;
  fprintf(stderr, "arctag-serve %s: model %s (%s) loaded in %.0f ms; listening on %s:%d\n", ARC_DEPLOY_VERSION, tagsrv_model_id(), dir,
          http_now_ms() - t0, host, c.port);
  return http_serve(&c);
}
