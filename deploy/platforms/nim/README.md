# NVIDIA NIM

**Status: blocked, CPU-only.** NIMs are GPU-accelerated containers. Arc Tagger has no GPU code: the native backend is
plain C (AVX2+FMA / NEON, one thread per handle) and the optional ONNX Runtime backend uses the CPU provider. At about
10 M parameters and a few ms per word on one core, a GPU would not pay off.

Already NIM-shaped: `GET /v1/health/live`, `/v1/health/ready`, `/v1/models`, environment configuration, one port,
non-root, no network at run time.

The realistic NVIDIA route is as a CPU component of a GPU Arc Transcriber NIM, which depends on the GPU path in
[arc_transcriber `deploy/platforms/nim/`](https://github.com/WhissleAI/arc_transcriber/blob/main/deploy/platforms/nim/README.md).
A listing would also need Whissle's commercial terms (CC BY-NC 4.0 today) and NVIDIA's partner process.
