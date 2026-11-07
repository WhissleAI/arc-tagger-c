/*
 * arctag - C API for the Arc Tagger (text-only streaming tagger: entities, numbers, PII, profanity/toxicity spans,
 * events, utterance intents / dialogue acts / safety / emotion, content state).
 *
 * All returned strings are malloc'd UTF-8 JSON; release them with arctag_free_str(). A handle is not thread-safe:
 * use one handle per thread (or lock around calls). See README.md for the JSON schema.
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

typedef struct arctag arctag_t;

/* Load a model directory containing arctag.onnx, arctag_meta.json and tokenizer.json.
 * Returns NULL on failure and writes a message into err (if err != NULL). */
ARCTAG_API arctag_t *arctag_create(const char *model_dir, char *err, size_t errlen);

/* Options (return 0 on success, -1 on an unknown key / bad value):
 *   "domain"     profile: auto (default) | talk | meeting | court | assistant | sports | news, or a fine domain
 *                (nba nhl nfl soccer cricket court_appellate court_trial assistant_noisy stream_chat), or "ui:fine"
 *   "asr"        word source the text comes from (names in arctag_meta.json "asr_sources"; default: meta "default_asr")
 *   "threads"    ONNX Runtime intra-op threads (0 = ORT default)
 *   "max_ctx"    words per model pass for arctag_tag_text (default 2048 = the model's context window)
 *   "stream_ctx" words of context re-run by arctag_poll (default 256)
 * Changing domain / asr resets the stream. */
ARCTAG_API int arctag_set_option(arctag_t *h, const char *key, const char *value);

/* Tag a whole text. Words = whitespace split; a word ending in . ? ! closes an utterance.
 * Returns JSON: {"profile","n_words","words":[...],"spans":[...],"events":[...],"utterances":[...]} or NULL on error. */
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

/* Run the model over the newest stream_ctx words and return what changed:
 * {"n_words", "a0", "spans": [spans ending after a0; replace your spans with end > a0], "events": [new events],
 *  "utterances": [read-outs of utterances closed since the last poll], "current": running read-out or null}. */
ARCTAG_API char *arctag_poll(arctag_t *h);

/* Forget the stream (words, events, utterances). */
ARCTAG_API void arctag_reset(arctag_t *h);

/* Last error message of this handle ("" if none). */
ARCTAG_API const char *arctag_last_error(const arctag_t *h);

ARCTAG_API void arctag_free_str(char *s);
ARCTAG_API void arctag_destroy(arctag_t *h);
ARCTAG_API const char *arctag_version(void);

#ifdef __cplusplus
}
#endif
#endif /* ARCTAG_H */
