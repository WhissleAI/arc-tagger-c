/* Byte-level BPE (GPT-2 style) tokenizer reading a HuggingFace tokenizer.json (model.type = BPE, pre_tokenizer = ByteLevel). */
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct arctok arctok_t;

arctok_t *arctok_load(const char *tokenizer_json_path, char *err, size_t errlen);
void arctok_free(arctok_t *t);
/* Encode `text` exactly like tokenizers' Tokenizer.encode(text, add_special_tokens=False).ids.
   Writes up to `cap` ids, returns the total number of ids (may exceed cap), or -1 on error. */
int arctok_encode(const arctok_t *t, const char *text, int32_t *ids, int cap);

/* UTF-8 helpers shared with the library */
int utf8_decode(const char *s, size_t n, size_t *i, uint32_t *cp);   /* returns 0 on success, advances *i */
int uni_is_letter(uint32_t cp);
int uni_is_number(uint32_t cp);
int uni_is_space(uint32_t cp);
int uni_is_digit(uint32_t cp);                                     /* = Python str.isdigit() for one character */
