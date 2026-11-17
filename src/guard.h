/* Kids / game guard layer (domain "kids" / "game", or option guard=on): deterministic checks on top of the model's
 * read-outs for short spoken turns. See docs/KIDS.md. Internal header. */
#ifndef ARCTAG_GUARD_H
#define ARCTAG_GUARD_H

#include "cJSON.h"

/* profanity lexicon: 2 = strong, 1 = mild, -1 = minced oath ("dang", "heck": never profanity), 0 = not listed */
int guard_lex(const char *word);

/* Profanity spans of words [a, b) in the result's span list (absolute word indices):
 * model profanity spans are re-typed by the lexicon severity of their words, dropped when they cover only minced oaths
 * or only unlisted words below model_min; listed words no model span covers get a span with src "lexicon".
 * prof_thr is reported as the lexicon spans' "thr". The list stays sorted by (start, -end). */
void guard_spans(cJSON *spans, const char *const *words, int a, int b, double model_min, double prof_thr);

/* The utterance safety list for words [a, b): `all` holds every valid safety.* utterance column of the model
 * ({tag, p, thr, on}); spans is the result's span list (after guard_spans). cmd_ctx = the app says the turn is a
 * command on a game screen. Returns the new list (caller owns it). */
cJSON *guard_utt(const char *const *words, int a, int b, const cJSON *spans, const cJSON *all, int cmd_ctx, double prof_thr);

#endif
