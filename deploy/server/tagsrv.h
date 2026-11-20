/* tagsrv.h: Arc Tagger as a service. A pool of arctag handles for request / response calls, one handle per realtime
 * session, the request formats of POST /v1/tags (text, words, transcript segments, batches) and the WS /v1/realtime
 * protocol. Used by the tagger's own server (main.c) and by the combined transcriber + tagger image in
 * WhissleAI/arc_transcriber (deploy/, built with this file from a checkout of this repository).
 * Part of arc-tagger-c (Whissle Inc.), CC BY-NC 4.0 with the Whissle terms; see LICENSE. */
#ifndef ARCTAG_TAGSRV_H
#define ARCTAG_TAGSRV_H

#include <stddef.h>
#include "cJSON.h"
#include "http.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Settings, from the environment unless given:
 *   ARCTAG_PRECISION   fp32 (default, the reference) | int8
 *   ARCTAG_WORKERS     handles in the request pool (default 2)
 *   ARCTAG_MAX_STREAMS concurrent realtime sessions (default 16)
 *   ARCTAG_DOMAIN      default domain (default auto)
 *   ARCTAG_CONFIG      a configs/NAME.json file applied to every handle
 *   ARCTAG_MODEL_ID    the id reported by /v1/models (default arc-tagger)
 *   ARC_QUEUE_TIMEOUT_S how long a request waits for a free handle (default 60) */
int tagsrv_init(const char* model_dir, char* err, size_t err_len);
void tagsrv_shutdown(void);
int tagsrv_ready(void);
const char* tagsrv_model_id(void);
const char* tagsrv_model_dir(void);
/* arctag_info() of the pool's first handle, as a cJSON object (caller frees) */
cJSON* tagsrv_info(void);

/* One POST /v1/tags body (an object: text | words | segments | inputs[batch]) -> response object (caller frees).
 * *status: 200, 400 (bad request: the object is {"error": ...}), 503 (no free handle). */
cJSON* tagsrv_process(const cJSON* req, int* status);
/* Tag pre-split words (a JSON array of arctag word objects) with per-call options (domain, asr, emit, tags, redact,
 * thresholds, guard, ... as in a config; may be NULL). Returns the arctag output or {"error"} with *status != 200. */
cJSON* tagsrv_tag_words(const char* words_json, const cJSON* opts, int* status);
/* Arc Transcriber's verbose_json-like segments (each {text, start, end, speaker?, words?: [{word|text, start, end}]})
 * -> an arctag words array (eou on each segment's last word). Caller frees. NULL if none. */
cJSON* tagsrv_segments_to_words(const cJSON* segments);

/* Streaming handles (one per realtime session): NULL when ARCTAG_MAX_STREAMS are open (err says so). */
typedef struct tagsrv_stream tagsrv_stream;
tagsrv_stream* tagsrv_stream_open(const cJSON* opts, char* err, size_t err_len);
/* push (arctag_push_words JSON) then poll: the poll object, or {"error"} (caller frees) */
cJSON* tagsrv_stream_push(tagsrv_stream* s, const char* push_json);
int tagsrv_stream_apply(tagsrv_stream* s, const cJSON* opts, char* err, size_t err_len);
void tagsrv_stream_reset(tagsrv_stream* s);
void tagsrv_stream_close(tagsrv_stream* s);

/* WS /v1/realtime for the tagger (the request has been checked to be an upgrade). */
void tagsrv_ws_session(http_req* r);

#ifdef __cplusplus
}
#endif
#endif
