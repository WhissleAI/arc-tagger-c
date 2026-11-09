/* atn: the Arc Tagger forward pass in plain C. One recurrent step per word, weights memory-mapped from a flat .atw file, one
 * thread, no ONNX Runtime. arctag.c is the only intended user; tests and tools/export_model.py's parity check also use it.
 *
 * A model (atn_model) is immutable after atn_open and may be shared by any number of threads. A state (atn) belongs to one
 * thread at a time. There is no global mutable state in this file's forward pass. */
#ifndef ATN_H
#define ATN_H
#include <stddef.h>

#define ATN_NOUT 22
/* Words of attention memory needed beyond the window to serve a rewind: atn_new's ctx_cap must be at least win + this if
 * snapshots are on. Rewinds reach back at most SNAP_EVERY * SNAP_KEEP + SNAP_EVERY = 80 words (see atn.c). */
#define ATN_REWIND_WORDS 96

typedef struct atn_model atn_model;
typedef struct atn atn;
typedef struct { int pcs[8]; int npc; int wfl[5]; int feats[8]; int src; int spk; int ust; } atn_word;

/* Map and fully validate a weights file: magic, version, sizes, checksums, every tensor's bounds, the expected shape of every
 * table and the range of every index table. Returns NULL and a message in err on any problem. */
atn_model *atn_open(const char *path, char *err, size_t errlen);
/* The same validation over a caller-owned buffer, which must outlive the model and be aligned to 16 bytes. verify_crc = 0 skips
 * only the payload checksum (the fuzz test uses it to reach the deeper checks quickly). */
atn_model *atn_open_mem(const void *buf, size_t size, int verify_crc, char *err, size_t errlen);
void atn_close(atn_model *m);

int atn_out_dim(const atn_model *m, int o);     /* columns of output o (frame span_beg span_in span_end ev_at ev_imm ev_rec utt speaker item, sh_*, ver_m, ver_e) */
int atn_out_total(const atn_model *m);          /* sum of the columns */
const char *atn_out_name(int o);
int atn_num_domains(const atn_model *m);
const char *atn_weights_kind(const atn_model *m);  /* "fp32" or "int8", as written by the exporter */
const char *atn_simd_name(void);                /* the kernels compiled in: "neon", "avx2" or "scalar" */
const char *atn_gemm_name(void);                /* "accelerate" or "builtin" */

/* ctx_cap: words of attention memory kept. win: words attended to (older keys ignored). Returns NULL if out of memory or dom is
 * out of range. */
atn *atn_new(atn_model *m, int dom, int ctx_cap, int win);
void atn_free(atn *a);
int atn_set_domain(atn *a, int dom);            /* also resets; -1 if dom is out of range */
void atn_reset(atn *a);                         /* back to word 0 */
void atn_restart(atn *a, int pos);              /* forget everything; the next word stepped is word pos, with a fresh state */
int atn_pos(const atn *a);                      /* next word index */
int atn_first(const atn *a);                    /* the word the current state began at */
/* one word: writes atn_out_total() floats (the outputs, concatenated) */
void atn_step(atn *a, const atn_word *w, float *out);
/* T words (same results as T steps within rounding; matmuls batched). out: [T, total] */
void atn_steps(atn *a, const atn_word *w, int T, float *out);
/* Revision snapshots (default on). Off: one-shot calls use full blocks and no snapshot memory. */
void atn_set_snapshots(atn *a, int on);
/* Forget words >= k: restores the newest snapshot at or before k and returns its position p <= k; the caller re-steps words
 * p..k-1. Returns -1, changing nothing, if no snapshot reaches back that far; the caller then uses atn_restart. */
int atn_rewind(atn *a, int k);
#endif
