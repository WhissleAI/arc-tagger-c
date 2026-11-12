/* config files and runtime options: every configs/*.json loads (with the fp32 graph) and is reflected by arctag_info;
 * bad configs are rejected with a message; emit / tags filters, redaction and threshold overrides change the output the
 * documented way.   test_config MODEL_DIR CONFIG.json... */
#include "arctag.h"
#include "cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static cJSON *tag(arctag_t *h, const char *text) { char *s = arctag_tag_text(h, text); cJSON *j = s ? cJSON_Parse(s) : NULL; arctag_free_str(s); return j; }
static int count_prefix(cJSON *arr, const char *key, const char *pre) {
    int n = 0; for (cJSON *x = arr ? arr->child : NULL; x; x = x->next) if (!strncmp(cJSON_GetStringValue(cJSON_GetObjectItem(x, key)), pre, strlen(pre))) n++;
    return n;
}

static const char *TEXT = "Hi, this is Maria Lopez calling about my order. My phone number is 555 867 5309 and the order was placed on March 3rd. Can you tell me when it will ship?";

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL_DIR [CONFIG.json ...]\n", argv[0]); return 2; }
    const char *md = argv[1]; char err[512];
    /* 1. every config file loads; info reflects domain / emit / redaction */
    for (int i = 2; i < argc; i++) {
        arctag_t *h = arctag_create_from_config(md, argv[i], err, sizeof err);
        if (!h && strstr(err, "arctag_int8.onnx")) {                         /* int8 graph not in the model dir: load fp32 */
            fprintf(stderr, "note: %s needs arctag_int8.onnx (%s); checking it with fp32\n", argv[i], err);
            FILE *f = fopen(argv[i], "rb"); char buf[1 << 16]; size_t n = fread(buf, 1, sizeof buf - 1, f); fclose(f); buf[n] = 0;
            cJSON *c = cJSON_Parse(buf); cJSON_ReplaceItemInObject(c, "precision", cJSON_CreateString("fp32"));
            char *cs = cJSON_PrintUnformatted(c); cJSON_Delete(c);
            h = arctag_create(md, err, sizeof err);
            CHECK(h && !arctag_apply_config(h, cs), "%s: %s", argv[i], h ? arctag_last_error(h) : err);
            free(cs);
        } else CHECK(h != NULL, "%s: %s", argv[i], err);
        if (!h) continue;
        char *info = arctag_info(h); cJSON *in = cJSON_Parse(info);
        CHECK(in && cJSON_GetObjectItem(in, "profile"), "%s: info", argv[i]);
        cJSON *r = tag(h, TEXT);
        CHECK(r && cJSON_GetArraySize(cJSON_GetObjectItem(r, "utterances")) == 3, "%s: tag_text", argv[i]);
        printf("config %-60s ok: %s\n", argv[i], info);
        cJSON_Delete(r); cJSON_Delete(in); arctag_free_str(info); arctag_destroy(h);
    }
    arctag_t *h = arctag_create(md, err, sizeof err);
    if (!h) { fprintf(stderr, "%s\n", err); return 1; }
    /* 2. bad configs */
    CHECK(arctag_apply_config(h, "{\"domian\": \"talk\"}") != 0 && strstr(arctag_last_error(h), "domian"), "unknown key accepted");
    CHECK(arctag_apply_config(h, "{\"domain\": \"mars\"}") != 0, "unknown domain accepted");
    CHECK(arctag_apply_config(h, "{\"emit\": [\"spans\"]}") != 0, "unknown section accepted");
    CHECK(arctag_apply_config(h, "{\"thresholds\": {\"x@nope\": 0.5}}") != 0, "unknown read-out accepted");
    CHECK(arctag_apply_config(h, "{\"redact\": {\"mode\": \"mask\", \"policy\": {\"phone\": \"blur\"}}}") != 0, "bad policy accepted");
    CHECK(arctag_apply_config(h, "[1, 2]") != 0, "array accepted");
    CHECK(arctag_set_option(h, "precision", "fp16") != 0, "fp16 accepted");
    /* a failed precision switch keeps the loaded session usable */
    arctag_options o; arctag_options_init(&o); o.model_file = "does_not_exist.onnx";
    CHECK(arctag_create_ex(md, &o, err, sizeof err) == NULL, "missing model_file accepted");
    /* 3. defaults: everything, no redacted_text */
    cJSON *base = tag(h, TEXT);
    CHECK(base && !cJSON_GetObjectItem(base, "redacted_text"), "redacted_text without redaction");
    int n_pii = count_prefix(cJSON_GetObjectItem(base, "spans"), "type", "safety.pii.");
    int n_ent = count_prefix(cJSON_GetObjectItem(base, "spans"), "type", "entity.");
    CHECK(n_pii > 0 && n_ent > 0, "expected pii and entity spans (%d, %d)", n_pii, n_ent);
    cJSON *u0 = cJSON_GetArrayItem(cJSON_GetObjectItem(base, "utterances"), 0);
    CHECK(u0 && cJSON_GetObjectItem(u0, "valence") && cJSON_GetObjectItem(u0, "intensity"), "valence / intensity read-outs");
    /* 4. emit + tags */
    CHECK(!arctag_apply_config(h, "{\"emit\": [\"pii\", \"intent\"]}"), "%s", arctag_last_error(h));
    cJSON *r = tag(h, TEXT);
    CHECK(count_prefix(cJSON_GetObjectItem(r, "spans"), "type", "entity.") == 0 && count_prefix(cJSON_GetObjectItem(r, "spans"), "type", "safety.pii.") == n_pii, "emit spans");
    u0 = cJSON_GetArrayItem(cJSON_GetObjectItem(r, "utterances"), 0);
    CHECK(cJSON_GetObjectItem(u0, "intent") && !cJSON_GetObjectItem(u0, "act") && !cJSON_GetObjectItem(u0, "state"), "emit read-outs");
    CHECK(cJSON_GetArraySize(cJSON_GetObjectItem(r, "events")) == 0, "emit events");
    cJSON_Delete(r);
    CHECK(!arctag_apply_config(h, "{\"emit\": [], \"tags\": [\"safety.pii.phone\"]}"), "%s", arctag_last_error(h));
    r = tag(h, TEXT);
    CHECK(cJSON_GetArraySize(cJSON_GetObjectItem(r, "spans")) == count_prefix(cJSON_GetObjectItem(r, "spans"), "type", "safety.pii.phone"), "tags filter");
    cJSON_Delete(r);
    CHECK(!arctag_apply_config(h, "{\"emit\": [], \"tags\": []}"), "%s", arctag_last_error(h));
    /* 5. redaction */
    CHECK(!arctag_set_option(h, "redact", "phone=mask"), "%s", arctag_last_error(h));
    r = tag(h, TEXT);
    const char *red = cJSON_GetStringValue(cJSON_GetObjectItem(r, "redacted_text"));
    CHECK(red && strstr(red, "[PHONE]") && !strstr(red, "5309") && strstr(red, "Maria"), "phone-only redaction: %s", red ? red : "(null)");
    cJSON_Delete(r);
    CHECK(!arctag_set_option(h, "redact", "mask"), "%s", arctag_last_error(h));
    r = tag(h, TEXT); red = cJSON_GetStringValue(cJSON_GetObjectItem(r, "redacted_text"));
    CHECK(red && strstr(red, "[PHONE]") && strstr(red, "[NAME]"), "default redaction: %s", red ? red : "(null)");
    cJSON_Delete(r);
    CHECK(!arctag_set_option(h, "redact", "off"), "redact off");
    /* 6. threshold overrides: a family wildcard at 1.01 removes the family (span head + span_in), "" removes the override */
    CHECK(!arctag_set_option(h, "thr:safety.pii.*", "1.01"), "%s", arctag_last_error(h));
    r = tag(h, TEXT);
    CHECK(count_prefix(cJSON_GetObjectItem(r, "spans"), "type", "safety.pii.") == 0, "pii wildcard 1.01");
    cJSON_Delete(r);
    CHECK(!arctag_set_option(h, "thr:safety.pii.*", ""), "%s", arctag_last_error(h));
    CHECK(!arctag_set_option(h, "thr:head:entity.mention", "1.01"), "%s", arctag_last_error(h));
    r = tag(h, TEXT);
    CHECK(count_prefix(cJSON_GetObjectItem(r, "spans"), "type", "entity.mention.") == 0 &&
          count_prefix(cJSON_GetObjectItem(r, "spans"), "type", "safety.pii.") == n_pii, "head override");
    cJSON_Delete(r);
    CHECK(!arctag_set_option(h, "thr:intent.*@utt", "0"), "%s", arctag_last_error(h));
    r = tag(h, TEXT); u0 = cJSON_GetArrayItem(cJSON_GetObjectItem(r, "utterances"), 0);
    int all_on = 1;
    for (cJSON *x = cJSON_GetObjectItem(u0, "intent")->child; x; x = x->next)
        if (!strncmp(cJSON_GetStringValue(cJSON_GetObjectItem(x, "tag")), "intent.", 7) && !cJSON_IsTrue(cJSON_GetObjectItem(x, "on"))) all_on = 0;
    CHECK(all_on, "utt override at 0");
    cJSON_Delete(r);
    CHECK(!arctag_set_option(h, "thr_reset", ""), "thr_reset");
    r = tag(h, TEXT);
    CHECK(r && cJSON_GetArraySize(cJSON_GetObjectItem(r, "spans")) == cJSON_GetArraySize(cJSON_GetObjectItem(base, "spans")), "thr_reset restores");
    cJSON_Delete(r);
    /* 7. domain aliases + streaming profile in poll */
    CHECK(!arctag_set_option(h, "domain", "soccer"), "%s", arctag_last_error(h));
    arctag_push_words(h, "{\"words\": [\"goal!\"], \"endpoint\": true}");
    char *p = arctag_poll(h); cJSON *pj = cJSON_Parse(p); arctag_free_str(p);
    CHECK(pj && !strcmp(cJSON_GetStringValue(cJSON_GetObjectItem(pj, "profile")), "sports:soccer"), "poll profile");
    cJSON_Delete(pj);
    cJSON_Delete(base);
    arctag_destroy(h);
    printf("config tests: %d failures\n", fails);
    return fails ? 1 : 0;
}
