# NVIDIA NIM

**Status: not a NIM candidate on its own.** NIMs are GPU-accelerated inference containers. Arc Tagger runs on CPU:
the native backend is plain C (AVX2+FMA / NEON kernels, one thread per handle) and the optional ONNX Runtime backend
uses the CPU provider; there is no GPU code. The model has about 10 M parameters and streams at a few ms per word on
one core, so a GPU would not pay for itself.

Already NIM-shaped: `GET /v1/health/live`, `GET /v1/health/ready`, `GET /v1/models`, environment configuration, one
port, non-root, no network at run time.

The realistic NVIDIA route is as a component of a GPU Arc Transcriber NIM (the tagger running on the same container's
CPU after a GPU transcriber), which depends on the GPU path described in WhissleAI/arc_transcriber
`deploy/platforms/nim/`. A listing would also need Whissle's commercial terms (Arc Tagger is CC BY-NC 4.0 today) and
NVIDIA's partner process (NGC organisation, base images, security scanning).
