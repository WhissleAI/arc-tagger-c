# arctag-serve source

The C source of the hosted service. Usage, routes and configuration: [deploy/README.md](../README.md).

| file | what |
|---|---|
| `main.c` | routes, SageMaker and Snowflake shapes, start-up |
| `tagsrv.c`, `tagsrv.h` | handle pool, `/v1/tags` request formats, the realtime protocol; also linked into Arc Transcriber's combined server |
| `http.c`, `http.h` | HTTP/1.1 and WebSocket server (keep-alive, limits, access log) |

```bash
make -C deploy -j2                                  # deploy/build/arctag-serve
deploy/build/arctag-serve --port 8080 --model deploy/weights/arc-tagger
```
