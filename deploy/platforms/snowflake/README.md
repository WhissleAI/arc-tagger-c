# Snowflake Snowpark Container Services (SPCS)

The **bundled** image runs as an SPCS service; SQL calls it through a **service function**, which POSTs batches of
rows (`{"data": [[row, text, domain?, options?], ...]}`); arctag-serve answers `{"data": [[row, result], ...]}` at
`/v1/snowflake/tags`. No external access integration is needed.

```bash
deploy/fetch_weights.sh
docker buildx build --platform linux/amd64 -f deploy/Dockerfile --target bundled -t arc-tagger:0.1.0-v0.4.1 --load .   # SPCS: amd64
docker tag arc-tagger:0.1.0-v0.4.1 <org>-<account>.registry.snowflakecomputing.com/arc_db/arc/images/arc-tagger:0.1.0-v0.4.1
docker push <org>-<account>.registry.snowflakecomputing.com/arc_db/arc/images/arc-tagger:0.1.0-v0.4.1
```

Then [`setup.sql`](setup.sql) with [`service.yaml`](service.yaml), and:

```sql
SELECT id, arc_tags(transcript) AS t FROM calls;
SELECT id, s.value:type::string AS pii, s.value:text::string AS span
FROM (SELECT id, arc_tags_domain(transcript, 'talk') AS t FROM calls), LATERAL FLATTEN(t:spans) s
WHERE s.value:family = 'pii';
SELECT arc_tags_opts(note, 'talk', '{"redact": "mask"}'):redacted_text::string FROM notes;
```

One replica handles `ARCTAG_WORKERS` rows at a time (one core each); text rows are small, so `MAX_BATCH_ROWS` can be
large.
