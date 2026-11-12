/* onnx_file: a small reader for the parts of an .onnx file the native backend needs: the graph's initializers, its nodes
 * (operator, inputs, outputs, Constant values, integer attributes) and the shapes of its outputs. It reads the protobuf
 * wire format directly; no protobuf library. Every length and offset is checked against the buffer. */
#ifndef ONNX_FILE_H
#define ONNX_FILE_H
#include <stddef.h>
#include <stdint.h>

/* ONNX TensorProto data types used here */
enum { OX_FLOAT = 1, OX_UINT8 = 2, OX_INT8 = 3, OX_INT32 = 6, OX_INT64 = 7, OX_BOOL = 9, OX_DOUBLE = 11 };

#define OX_MAX_DIMS 8

typedef struct {
    const char *name;
    int dt, nd;
    int64_t d[OX_MAX_DIMS];
    int64_t count;                 /* number of elements */
    const uint8_t *data; size_t len;
    int enc;                       /* 0 raw_data, 1 packed float_data, 2 packed varints (int32_data / int64_data), 3 double_data */
} ox_tensor;

typedef struct {
    const char *op;
    int nin, nout;
    const char **in, **out;
    int64_t axis; int has_axis;
    ox_tensor value; int has_value;  /* Constant */
} ox_node;

typedef struct {
    const char *name; int nd; int64_t d[OX_MAX_DIMS];   /* d < 0: symbolic */
} ox_output;

typedef struct {
    uint8_t *buf; size_t size;     /* the file */
    ox_tensor *init; int ninit;
    ox_node *node; int nnode;
    ox_output *out; int nout;
    char *pool; size_t pool_used, pool_cap;
    const char **slots; size_t slots_used, slots_cap;
} ox_graph;

/* Read and parse a file. Returns 0, or -1 with a message in err. */
int ox_load(ox_graph *g, const char *path, char *err, size_t errlen);
/* Parse a caller-owned buffer (kept by g, not copied: it must outlive g; ox_free does not free it). */
int ox_parse(ox_graph *g, const uint8_t *buf, size_t size, char *err, size_t errlen);
void ox_free(ox_graph *g);

const ox_tensor *ox_init(const ox_graph *g, const char *name);
/* the value behind a name: an initializer, a Constant node's output, or either of those through Identity nodes */
const ox_tensor *ox_value(const ox_graph *g, const char *name);
const ox_node *ox_producer(const ox_graph *g, const char *name);
/* the only node of operator op that reads name, or NULL (none, or more than one) */
const ox_node *ox_only_consumer(const ox_graph *g, const char *name, const char *op);
int ox_node_index(const ox_graph *g, const ox_node *n);

/* Convert a tensor's elements to dst: float (OX_FLOAT), int8 (OX_INT8), uint8 (OX_UINT8 / OX_BOOL) or int64 (OX_INT64).
 * The source type must match the target's kind (float, or any integer type). Returns 0 or -1 (bad data). */
int ox_read(const ox_tensor *t, int dst_type, void *dst, int64_t count);
#endif
