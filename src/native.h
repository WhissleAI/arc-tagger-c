/* native: the Arc Tagger forward pass in plain C, one word at a time, with state kept between calls. It reads its weights
 * from the same arctag.onnx / arctag_int8.onnx files the ONNX Runtime backend uses. arctag.c is the only intended user.
 *
 * A model (nat_model) is immutable after nat_open and may be shared by any number of threads. A state (nat) belongs to one
 * thread at a time. There is no global mutable state. */
#ifndef ARCTAG_NATIVE_H
#define ARCTAG_NATIVE_H
#include <stddef.h>

#define NAT_NOUT 22
/* Words of history kept beyond the window so that a revision can rewind: nat_new's ctx_cap must be at least win + this
 * when snapshots are on. */
#define NAT_REWIND_WORDS 96

typedef struct nat_model nat_model;
typedef struct nat nat;
typedef struct { int pcs[8]; int npc; int wfl[5]; int feats[8]; int src; int spk; int ust; } nat_word;

/* Load and validate an .onnx file: every table the forward pass reads, its shape, and the range of every index table.
 * Tables stored as int8 with one scale per column stay int8 in memory; other quantised tables are converted to float.
 * Returns NULL and a message in err on any problem. */
nat_model *nat_open(const char *path, char *err, size_t errlen);
/* The same from a caller-owned buffer holding the file (not kept after the call). */
nat_model *nat_open_mem(const void *buf, size_t size, char *err, size_t errlen);
void nat_close(nat_model *m);

int nat_out_dim(const nat_model *m, int o);     /* columns of output o, in the graph's output order */
int nat_out_total(const nat_model *m);          /* sum of the columns */
const char *nat_out_name(int o);
int nat_num_domains(const nat_model *m);
const char *nat_weights_kind(const nat_model *m);  /* "fp32", "int8" (int8 tables kept) or "int8-folded" (converted to float) */
size_t nat_weights_bytes(const nat_model *m);    /* bytes of tables held in memory */
const char *nat_simd_name(void);                /* the kernels compiled in: "neon", "avx2" or "scalar" */

/* ctx_cap: words of history kept. win: words looked back at (older ones are ignored). NULL if out of memory or dom is out
 * of range. */
nat *nat_new(nat_model *m, int dom, int ctx_cap, int win);
void nat_free(nat *a);
int nat_set_domain(nat *a, int dom);            /* also resets; -1 if dom is out of range */
void nat_reset(nat *a);                         /* back to word 0 */
void nat_restart(nat *a, int pos);              /* forget everything; the next word stepped is word pos, with a fresh state */
int nat_pos(const nat *a);                      /* next word index */
int nat_first(const nat *a);                    /* the word the current state began at */
/* T words: writes T * nat_out_total() floats ([T, total], the outputs concatenated per word) */
void nat_steps(nat *a, const nat_word *w, int T, float *out);
/* Snapshots for revisions (default on). Off: one-shot passes use full blocks and no snapshot memory. */
void nat_set_snapshots(nat *a, int on);
/* Forget words >= k: restores the newest snapshot at or before k and returns its position p <= k; the caller re-steps words
 * p..k-1. Returns -1, changing nothing, if no snapshot reaches back that far; the caller then uses nat_restart. */
int nat_rewind(nat *a, int k);
#endif
