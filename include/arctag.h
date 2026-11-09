/*
 * arctag - C API for the Arc Tagger (text-only streaming tagger: entities, numbers, PII, profanity/toxicity spans,
 * events, utterance intents / dialogue acts / safety / emotion, content state).
 *
 * All returned strings are malloc'd UTF-8 JSON; release them with arctag_free_str(). A handle is not thread-safe, but handles
 * are independent: any number of them may run on different threads at once. See README.md for the JSON schema.
 *
 * The runtime is native C (src/atn.c): no ONNX Runtime at build or run time. Link Accelerate on Apple platforms.
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

/* Load a model directory containing arctag.atw (tools/export_model.py makes it from arctag.onnx), arctag_meta.json and
 * tokenizer.json. The weights file is memory-mapped and must not be changed while the handle is alive. A truncated, corrupt or
 * mismatched file is an error here, never a crash later.
 * Returns NULL on failure and writes a message into err (if err != NULL). */
ARCTAG_API arctag_t *arctag_create(const char *model_dir, char *err, size_t errlen);

/* Options (return 0 on success, -1 on an unknown key / bad value):
 *   "domain"     profile: auto (default) | talk | meeting | court | assistant | sports | news, or a fine domain
 *                (nba nhl nfl soccer cricket court_appellate court_trial assistant_noisy stream_chat), or "ui:fine"
 *   "asr"        word source the text comes from (names in arctag_meta.json "asr_sources"; default: meta "default_asr")
 *   "threads"    accepted for compatibility and ignored: a handle always runs on its caller's thread, alone. For parallelism use
 *                one handle per thread. $ARCTAG_THREADS is ignored too.
 *   "max_ctx"    words per model pass for arctag_tag_text / arctag_tag_words (default 2048); longer texts become independent chunks
 *   "stream_ctx" 16..2048 (default 256): the attention window of the stream, in words. The stream's global attention and mention
 *                memory look at the newest stream_ctx words; the recurrent state and pooled read-outs carry the whole session.
 *                Up to stream_ctx words, stream output equals arctag_tag_text on the same words (within rounding, given the same
 *                speaker flag; see below). Past it, tag_text attends to all the words of its pass and the stream does not, so
 *                scores differ. Cost per word and memory do not grow with session length, only with stream_ctx.
 *                Changing it replays the held words (see arctag_push_words); events already returned are not repeated.
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
 * Word indices count the whole session, but only the newest words are held (stream_ctx + 272 to + 336 with the default metadata).
 * A revision costs about 1 ms per word re-run: within the last 48 words it is exact, deeper ones replay the held words from a fresh
 * state (exact while the session is still fully held, approximate after). k older than the oldest held word is an error.
 * Speaker labels: the model's "has speakers" input turns on at the first labelled word and stays on, without a replay, so a word's
 * scores never depend on later words. arctag_tag_text sets it for the whole pass (the graph's definition), so the two differ if labels
 * first appear mid-session. If labels are present from the first word they agree.
 * Returns 0 on success. -1 changes nothing and sets the error: a parse error, a revision too old, or more than 65536 words waiting
 * for arctag_poll. (Out of memory also returns -1; words before the failure may remain.) */
ARCTAG_API int arctag_push_words(arctag_t *h, const char *words_json);

/* Run the model over the words not run yet, keeping its state between calls (cost about 1 ms per new word at any session length),
 * and return what changed:
 * {"n_words", "a0", "spans": [spans ending after a0; replace your spans with end > a0], "events": [new events],
 *  "utterances": [read-outs of utterances closed since the last poll], "current": running read-out or null}. */
ARCTAG_API char *arctag_poll(arctag_t *h);

/* Forget the stream (words, events, utterances). Memory stays allocated for the next session. */
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
