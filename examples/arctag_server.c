/* arctag_server: a tiny single-threaded JSON server around one arctag handle (one stream).
 *
 *   arctag_server -m MODEL_DIR --stdin            JSON lines on stdin -> one JSON line per request on stdout
 *        {"op":"tag","text":"..."}  {"op":"tag_words","words":[...]}  {"op":"push","words":[...],"from":k,"endpoint":true}
 *        {"op":"poll"}  {"op":"reset"}  {"op":"set","key":"domain","value":"meeting"}
 *   arctag_server -m MODEL_DIR --port 8090        HTTP on 127.0.0.1 (use --host 0.0.0.0 to listen on all interfaces)
 *        POST /tag    body: raw text, or {"text": "...", "domain": "..."}
 *        POST /push   body: the arctag_push_words JSON        POST|GET /poll        POST /reset        GET /health
 */
#include "arctag.h"
#include "cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define strdup _strdup
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET sock_t;
#define CLOSESOCK closesocket
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int sock_t;
#define CLOSESOCK close
#endif

static char *err_json(const char *msg) {
    cJSON *o = cJSON_CreateObject(); cJSON_AddStringToObject(o, "error", msg ? msg : "error");
    char *s = cJSON_PrintUnformatted(o); cJSON_Delete(o); return s;
}

/* one request object -> malloc'd JSON response */
static char *handle(arctag_t *h, cJSON *req) {
    cJSON *op = cJSON_GetObjectItem(req, "op");
    const char *o = cJSON_IsString(op) ? op->valuestring : "";
    cJSON *dom = cJSON_GetObjectItem(req, "domain");
    if (cJSON_IsString(dom) && arctag_set_option(h, "domain", dom->valuestring)) return err_json(arctag_last_error(h));
    if (!strcmp(o, "tag")) {
        cJSON *t = cJSON_GetObjectItem(req, "text");
        if (!cJSON_IsString(t)) return err_json("missing text");
        char *r = arctag_tag_text(h, t->valuestring);
        return r ? r : err_json(arctag_last_error(h));
    }
    if (!strcmp(o, "tag_words")) {
        char *w = cJSON_PrintUnformatted(cJSON_GetObjectItem(req, "words"));
        char *r = w ? arctag_tag_words(h, w) : NULL; free(w);
        return r ? r : err_json(arctag_last_error(h));
    }
    if (!strcmp(o, "push")) {
        cJSON *c = cJSON_Duplicate(req, 1); cJSON_DeleteItemFromObject(c, "op"); cJSON_DeleteItemFromObject(c, "domain");
        char *w = cJSON_PrintUnformatted(c); cJSON_Delete(c);
        int rc = arctag_push_words(h, w); free(w);
        return rc ? err_json(arctag_last_error(h)) : arctag_poll(h);
    }
    if (!strcmp(o, "poll")) { char *r = arctag_poll(h); return r ? r : err_json(arctag_last_error(h)); }
    if (!strcmp(o, "reset")) { arctag_reset(h); return strdup("{\"ok\":true}"); }
    if (!strcmp(o, "set")) {
        cJSON *k = cJSON_GetObjectItem(req, "key"), *v = cJSON_GetObjectItem(req, "value");
        if (!cJSON_IsString(k) || !cJSON_IsString(v)) return err_json("set needs key and value strings");
        return arctag_set_option(h, k->valuestring, v->valuestring) ? err_json(arctag_last_error(h)) : strdup("{\"ok\":true}");
    }
    return err_json("unknown op (tag | tag_words | push | poll | reset | set)");
}

static int run_stdin(arctag_t *h) {
    size_t cap = 1 << 16; char *line = (char *)malloc(cap);
    for (;;) {
        size_t n = 0; int c;
        while ((c = getchar()) != EOF && c != '\n') { if (n + 1 >= cap) { cap *= 2; line = (char *)realloc(line, cap); } line[n++] = (char)c; }
        if (c == EOF && n == 0) break;
        line[n] = 0;
        if (!n) continue;
        cJSON *req = cJSON_Parse(line);
        char *out = req ? handle(h, req) : err_json("invalid JSON");
        cJSON_Delete(req);
        printf("%s\n", out); fflush(stdout); free(out);
    }
    free(line);
    return 0;
}

static void send_all(sock_t s, const char *b, size_t n) {
    while (n) { int k = (int)send(s, b, (int)n, 0); if (k <= 0) return; b += k; n -= (size_t)k; }
}

static void reply(sock_t c, int code, const char *body) {
    char hdr[256];
    int n = snprintf(hdr, sizeof hdr, "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
                     code, code == 200 ? "OK" : code == 404 ? "Not Found" : "Bad Request", strlen(body));
    send_all(c, hdr, (size_t)n); send_all(c, body, strlen(body));
}

static int run_http(arctag_t *h, const char *host, int port) {
#ifdef _WIN32
    WSADATA wd; WSAStartup(MAKEWORD(2, 2), &wd);
#endif
    sock_t s = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof one);
    struct sockaddr_in a; memset(&a, 0, sizeof a);
    a.sin_family = AF_INET; a.sin_port = htons((unsigned short)port); inet_pton(AF_INET, host, &a.sin_addr);
    if (bind(s, (struct sockaddr *)&a, sizeof a) || listen(s, 16)) { perror("bind/listen"); return 1; }
    fprintf(stderr, "arctag_server %s listening on http://%s:%d\n", arctag_version(), host, port);
    for (;;) {
        sock_t c = accept(s, NULL, NULL);
        if (c < 0) continue;
        size_t cap = 1 << 16, n = 0; char *buf = (char *)malloc(cap + 1);
        long clen = -1; char *body = NULL;
        for (;;) {                                                     /* read headers + Content-Length body */
            if (n == cap) { cap *= 2; buf = (char *)realloc(buf, cap + 1); }
            int k = (int)recv(c, buf + n, (int)(cap - n), 0);
            if (k <= 0) break;
            n += (size_t)k; buf[n] = 0;
            char *e = strstr(buf, "\r\n\r\n");
            if (e && clen < 0) {
                body = e + 4; clen = 0;
                for (char *p = buf; p < e; p++)
                    if (!strncmp(p, "\r\nContent-Length:", 17) || !strncmp(p, "\r\ncontent-length:", 17)) clen = atol(p + 17);
            }
            if (body && (long)(n - (size_t)(body - buf)) >= clen) break;
        }
        if (!body) { CLOSESOCK(c); free(buf); continue; }
        body[clen > 0 ? clen : 0] = 0;
        char method[8] = {0}, path[256] = {0};
        sscanf(buf, "%7s %255s", method, path);
        char *out = NULL; int code = 200;
        cJSON *req = NULL;
        if (!strcmp(path, "/health")) out = strdup("{\"ok\":true}");
        else if (!strcmp(path, "/tag")) {
            req = cJSON_Parse(body);
            if (!cJSON_IsObject(req)) { cJSON_Delete(req); req = cJSON_CreateObject(); cJSON_AddStringToObject(req, "text", body); }
            cJSON_AddStringToObject(req, "op", "tag");
        } else if (!strcmp(path, "/push")) {
            cJSON *w = cJSON_Parse(body);
            if (cJSON_IsArray(w)) { req = cJSON_CreateObject(); cJSON_AddItemToObject(req, "words", w); }
            else req = w;
            if (req) cJSON_AddStringToObject(req, "op", "push");
        } else if (!strcmp(path, "/poll")) { req = cJSON_CreateObject(); cJSON_AddStringToObject(req, "op", "poll"); }
        else if (!strcmp(path, "/reset")) { req = cJSON_CreateObject(); cJSON_AddStringToObject(req, "op", "reset"); }
        else { code = 404; out = err_json("endpoints: POST /tag, POST /push, GET /poll, POST /reset, GET /health"); }
        if (!out) {
            if (req) { out = handle(h, req); if (!strncmp(out, "{\"error\"", 8)) code = 400; }
            else { code = 400; out = err_json("invalid JSON body"); }
        }
        cJSON_Delete(req);
        reply(c, code, out);
        free(out); free(buf);
        CLOSESOCK(c);
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *model = getenv("ARCTAG_MODEL_DIR"), *host = "127.0.0.1", *domain = NULL;
    int port = 0, use_stdin = 0;
    if (!model) model = "model";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) domain = argv[++i];
        else if (!strcmp(argv[i], "--stdin")) use_stdin = 1;
        else { fprintf(stderr, "usage: %s [-m MODEL_DIR] [-d DOMAIN] (--stdin | --port N [--host ADDR])\n", argv[0]); return 2; }
    }
    char err[512];
    arctag_t *h = arctag_create(model, err, sizeof err);
    if (!h) { fprintf(stderr, "arctag_create(%s): %s\n", model, err); return 1; }
    if (domain && arctag_set_option(h, "domain", domain)) { fprintf(stderr, "%s\n", arctag_last_error(h)); return 1; }
    int rc = (port > 0 && !use_stdin) ? run_http(h, host, port) : run_stdin(h);
    arctag_destroy(h);
    return rc;
}
