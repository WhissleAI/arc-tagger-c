# syntax=docker/dockerfile:1
# Arc Tagger as a service: arctag-serve (native C backend: no ONNX Runtime, no Python) on a distroless base.
# CPU only (the native backend has AVX2+FMA / NEON kernels; it has no GPU path), so there is no CUDA variant.
# x86-64 hosts need AVX2 + FMA (every current cloud instance type has them); arm64 uses NEON.
#
#   docker buildx build --platform linux/amd64,linux/arm64 -f deploy/Dockerfile --target runtime -t arc-tagger:0.1.0 .
#   deploy/fetch_weights.sh && docker buildx build ... --target bundled --build-arg WEIGHTS=deploy/weights/arc-tagger .
#
# A later "Pro" image: the same Dockerfile, --build-arg WEIGHTS=<pro model folder> --build-arg MODEL_ID=arc-tagger-pro.
# Part of arc-tagger-c (Whissle Inc.), CC BY-NC 4.0 with the Whissle terms; see LICENSE.

ARG BUILD_IMAGE=debian:bookworm-slim
ARG RUNTIME_IMAGE=gcr.io/distroless/cc-debian12:nonroot

FROM ${BUILD_IMAGE} AS build
RUN apt-get update && apt-get install -y --no-install-recommends gcc make libc6-dev binutils && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY include include
COPY src src
COPY third_party/cjson third_party/cjson
COPY deploy/Makefile deploy/Makefile
COPY deploy/server deploy/server
ARG VERSION=0.1.0
# CFLAGS / LDFLAGS from the environment: the Makefile appends its own flags (-std, -I, ...) to them
ENV CFLAGS="-O2 -fstack-protector-strong -D_FORTIFY_SOURCE=2" LDFLAGS="-Wl,-z,relro,-z,now"
RUN make -C deploy -j2 VERSION="${VERSION}" \
 && strip deploy/build/arctag-serve \
 && ./deploy/build/arctag-serve --version

FROM ${RUNTIME_IMAGE} AS runtime
ARG VERSION=0.1.0
LABEL org.opencontainers.image.title="arc-tagger" \
      org.opencontainers.image.description="Arc Tagger: streaming text tagger (entities, numbers, PII, safety, intent, affect) on CPU" \
      org.opencontainers.image.vendor="Whissle Inc." \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.licenses="CC-BY-NC-4.0" \
      org.opencontainers.image.source="https://github.com/WhissleAI/arc-tagger-c" \
      com.amazonaws.sagemaker.capabilities.accept-bind-to-port="true"
COPY --from=build /src/deploy/build/arctag-serve /opt/arc/bin/serve
COPY deploy/openapi.yaml /opt/arc/openapi.yaml
COPY LICENSE NOTICE /opt/arc/
ENV PORT=8080 HOST=0.0.0.0 ARCTAG_WORKERS=2 ARC_ACCESS_LOG=1
EXPOSE 8080
USER 65532:65532
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD ["/opt/arc/bin/serve", "--healthcheck"]
ENTRYPOINT ["/opt/arc/bin/serve"]

FROM runtime AS bundled
ARG WEIGHTS=deploy/weights/arc-tagger
ARG MODEL_ID=arc-tagger
COPY --chown=0:0 ${WEIGHTS} /opt/arc/model
ENV ARCTAG_MODEL_ID=${MODEL_ID}
