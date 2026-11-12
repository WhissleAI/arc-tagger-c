/*
 * arctag - C API for the Arc Tagger (text-only streaming tagger: entities, numbers, PII, profanity/toxicity spans,
 * events, utterance intents / dialogue acts / safety / emotion / arousal / valence / intensity, speaker roles, content state).
 *
 * All returned strings are malloc'd UTF-8 JSON; release them with arctag_free_str(). A handle is not thread-safe:
 * use one handle per thread (or lock around calls). See README.md and docs/USAGE.md for the JSON schema.
 */
#ifndef ARCTAG_H
#define ARCTAG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(ARCTAG_BUILD_SHARED)
#define ARCTAG_API __declspec(dllexport)
#elif defined(_WIN32) && defined(ARCTAG_USE_SHARED)
#define ARCTAG_API __declspec(dllimport)
#else
#define ARCTAG_API
#endif

#define ARCTAG_VERSION_MAJOR 0
#define ARCTAG_VERSION_MINOR 3
#define ARCTAG_VERSION_PATCH 0

typedef struct arctag arctag_t;

/* Model precision: which ONNX graph of the model directory is loaded. */
typedef enum {
    ARCTAG_FP32 = 0,   /* arctag.onnx (reference accuracy) */
    ARCTAG_INT8 = 1    /* arctag_int8.onnx (int8 weights: smaller file, close to fp32; see docs/int8.md) */
} arctag_precision_t;

/* What runs the model. Both read the same model files and give the same output (see README, "Backends"). */
typedef enum {
    ARCTAG_BACKEND_DEFAULT = 0,       /* $ARCTAG_BACKEND if set, else native */
    ARCTAG_BACKEND_NATIVE = 1,        /* plain C, one thread; streaming polls only compute new words */
    ARCTAG_BACKEND_ONNXRUNTIME = 2    /* ONNX Runtime (builds with ARCTAG_WITH_ORT, the default) */
} arctag_backend_t;

/* Load-time options for arctag_create_ex. Always initialise with arctag_options_init() first, then set fields:
 * new fields are only ever appended, and `size` tells the library which ones the caller knows about. */
typedef struct arctag_options {
    size_t size;                    /* = sizeof(arctag_options), set by arctag_options_init */
    arctag_precision_t precision;   /* ARCTAG_FP32 (default) or ARCTAG_INT8 */
    int threads;                    /* ONNX Runtime intra-op threads; 0 = $ARCTAG_THREADS or the ORT default (native: always 1) */
    const char *model_file;         /* optional ONNX file (name inside model_dir, or a path) instead of the precision's default */
    arctag_backend_t backend;       /* ARCTAG_BACKEND_DEFAULT (native unless $ARCTAG_BACKEND says otherwise) */
} arctag_options;

ARCTAG_API void arctag_options_init(arctag_options *opt);

/* Load a model directory containing arctag.onnx (and/or arctag_int8.onnx), arctag_meta.json and tokenizer.json.
 * arctag_create uses fp32 unless $ARCTAG_PRECISION=int8, and the native backend unless $ARCTAG_BACKEND=onnxruntime.
 * Returns NULL on failure and writes a message into err. */
ARCTAG_API arctag_t *arctag_create(const char *model_dir, char *err, size_t errlen);
ARCTAG_API arctag_t *arctag_create_ex(const char *model_dir, const arctag_options *opt, char *err, size_t errlen);

/* Load a model directory and apply a JSON config file (configs/NAME.json: precision, threads, domain, emitted families,
 * threshold overrides, redaction, ...). The config's precision / threads are used for the first session load.
 * model_dir NULL = the config's "model_dir" (relative to the config file) or "model". */
ARCTAG_API arctag_t *arctag_create_from_config(const char *model_dir, const char *config_path, char *err, size_t errlen);

/* Apply a JSON config (the text, not a path) to a handle; same keys as the config files. Returns 0 or -1. */
ARCTAG_API int arctag_apply_config(arctag_t *h, const char *config_json);

/* Options (return 0 on success, -1 on an unknown key / bad value):
 *   "domain"     profile: auto (default) | talk | meeting | court | assistant | sports | news, or a fine domain
 *                (nba nhl nfl soccer cricket court_appellate court_trial assistant_noisy stream_chat), or "ui:fine"
 *   "asr"        word source the text comes from (names in arctag_meta.json "asr_sources"; default "unknown")
 *   "backend"    native | onnxruntime (reloads the model)
 *   "precision"  fp32 | int8 (reloads the model)
 *   "threads"    ONNX Runtime intra-op threads (0 = ORT default; reloads the session). The native backend uses one thread.
 *   "max_ctx"    words per model pass for arctag_tag_text (default 2048 = the model's context limit)
 *   "stream_ctx" words of context arctag_poll looks back at (default 256)
 *   "stream_state" on | off (native backend; default on): keep the model state between polls so a poll only computes
 *                the new words. off = re-run the last stream_ctx words on every poll, as the onnxruntime backend does.
 *   "fine_auto"  on | off: pick the fine domain inside sports / court from lexicon cues (default on)
 *   "emit"       comma list of output sections (default all): entity number pii profanity toxicity events intent act
 *                safety emotion arousal valence intensity role item state
 *   "tags"       comma list of tag prefixes to keep in the output ("" = all)
 *   "redact"     off (default) | mask | a policy "phone=mask,name=mask,contact=off,..." (categories: phone email address
 *                card gov_id dob name handle contact profanity toxicity; mask = replace by [PHONE] etc., off / tag = keep)
 *   "thr:PATTERN" a threshold override: PATTERN = a tag, a prefix ending in '*', optionally "@readout"; "head:<family>"
 *                for the span-head families (entity.number entity.mention safety.pii safety.profanity safety.toxicity).
 *                A family wildcard ("safety.pii.*") also sets that family's span-head threshold. Value "" removes it.
 *   "thr_reset"  drop every threshold override
 * Changing domain / asr resets the stream. */
ARCTAG_API int arctag_set_option(arctag_t *h, const char *key, const char *value);

/* Tag a whole text. Words = whitespace split; a word ending in . ? ! closes an utterance.
 * Returns JSON: {"profile","n_words","words":[...],"spans":[...],"events":[...],"utterances":[...]} (+ "redacted_text"
 * when redaction is on) or NULL on error. */
ARCTAG_API char *arctag_tag_text(arctag_t *h, const char *text);

/* Tag pre-split words: JSON array of strings or word objects {"w","t0","t1","conf","spk","final"} plus optional
 * per-word "eou": true (utterance ends after this word). Same output as arctag_tag_text. */
ARCTAG_API char *arctag_tag_words(arctag_t *h, const char *words_json);

/* Streaming input. words_json is one of
 *   ["word", {"w": "word", "t0": 1.2, "t1": 1.5, "conf": 0.9, "spk": "A", "final": true}, ...]   append words
 *   {"words": [...], "from": k, "endpoint": true}    replace words[k:] (revision; default k = current length),
 *                                                     then optionally close the utterance
 *   {"endpoint": true}                               close the current utterance
 * Returns 0 on success, -1 on a parse error. */
ARCTAG_API int arctag_push_words(arctag_t *h, const char *words_json);

/* Bring the model up to date with the stream (the native backend computes only the words added or revised since the last
 * poll; the onnxruntime backend re-runs the newest stream_ctx words) and return what changed:
 * {"n_words", "a0", "profile", "spans": [spans ending after a0; replace your spans with end > a0], "events": [new events],
 *  "utterances": [read-outs of utterances closed since the last poll], "current": running read-out or null}
 * (+ "redacted_text" of words[a0:] when redaction is on). */
ARCTAG_API char *arctag_poll(arctag_t *h);

/* Forget the stream (words, events, utterances). */
ARCTAG_API void arctag_reset(arctag_t *h);

/* Handle description: {"version","precision","backend","model_file","profile","asr","threads","max_ctx","stream_ctx",...}. */
ARCTAG_API char *arctag_info(arctag_t *h);

/* Last error message of this handle ("" if none). */
ARCTAG_API const char *arctag_last_error(const arctag_t *h);

ARCTAG_API void arctag_free_str(char *s);
ARCTAG_API void arctag_destroy(arctag_t *h);
ARCTAG_API const char *arctag_version(void);

#ifdef __cplusplus
}
#endif
#endif /* ARCTAG_H */
