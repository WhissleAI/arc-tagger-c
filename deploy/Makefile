# arctag-serve: the Arc Tagger HTTP + WebSocket service, linked against the library's native backend (no ONNX Runtime,
# no Python). C99/C11, libc, libm, pthreads (and Accelerate on macOS, as the library's CMake build does).
#   make -C deploy             -> deploy/build/arctag-serve
#   make -C deploy run MODEL=model
# Part of arc-tagger-c (Whissle Inc.), CC BY-NC 4.0 with the Whissle terms; see LICENSE.
CC ?= cc
CFLAGS ?= -O2
B ?= build
ROOT := ..
CFLAGS += -std=gnu11 -Wall -Wno-unused-function -pthread -I$(ROOT)/include -I$(ROOT)/src -I$(ROOT)/third_party/cjson -Iserver
LDLIBS += -lm -pthread
ifeq ($(shell uname -s),Darwin)
LDLIBS += -framework Accelerate
endif
VERSION ?= 0.1.0+$(shell git -C $(ROOT) rev-parse --short HEAD 2>/dev/null || echo dev)
CFLAGS += -DARC_DEPLOY_VERSION=\"$(VERSION)\"

LIB_SRC := $(ROOT)/src/arctag.c $(ROOT)/src/guard.c $(ROOT)/src/native.c $(ROOT)/src/onnx_file.c $(ROOT)/src/tokenizer.c $(ROOT)/third_party/cjson/cJSON.c
SRV_SRC := server/http.c server/tagsrv.c server/main.c
LIB_OBJ := $(patsubst $(ROOT)/%.c,$(B)/lib/%.o,$(LIB_SRC))
SRV_OBJ := $(patsubst server/%.c,$(B)/srv/%.o,$(SRV_SRC))

all: $(B)/arctag-serve

$(B)/lib/%.o: $(ROOT)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@
$(B)/srv/%.o: server/%.c server/http.h server/tagsrv.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@
$(B)/arctag-serve: $(LIB_OBJ) $(SRV_OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

MODEL ?= ../model
PORT ?= 8080
run: $(B)/arctag-serve
	ARCTAG_MODEL_DIR=$(MODEL) PORT=$(PORT) $(B)/arctag-serve

clean:
	rm -rf $(B)
.PHONY: all run clean
