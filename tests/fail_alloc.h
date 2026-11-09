/* Force-included by the out-of-memory test: all allocations in the library go through these. */
#ifndef FAIL_ALLOC_H
#define FAIL_ALLOC_H
#include <stdlib.h>
#include <string.h>
#ifdef __cplusplus
extern "C" {
#endif
void *fi_malloc(size_t n);
void *fi_calloc(size_t n, size_t z);
void *fi_realloc(void *p, size_t n);
char *fi_strdup(const char *s);
void fi_arm(long nth);        /* the nth allocation from now fails; 0 disarms */
long fi_count(void);
#ifdef __cplusplus
}
#endif
#define malloc fi_malloc
#define calloc fi_calloc
#define realloc fi_realloc
#define strdup fi_strdup
#endif
