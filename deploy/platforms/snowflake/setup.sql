-- Arc Tagger on Snowpark Container Services.
CREATE DATABASE IF NOT EXISTS arc_db;
CREATE SCHEMA IF NOT EXISTS arc_db.arc;
CREATE IMAGE REPOSITORY IF NOT EXISTS arc_db.arc.images;
CREATE STAGE IF NOT EXISTS arc_db.arc.specs;
-- PUT file://deploy/platforms/snowflake/service.yaml @arc_db.arc.specs AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
CREATE COMPUTE POOL IF NOT EXISTS arc_tag_pool MIN_NODES = 1 MAX_NODES = 2 INSTANCE_FAMILY = CPU_X64_XS;
CREATE SERVICE IF NOT EXISTS arc_db.arc.arc_tagger_svc
  IN COMPUTE POOL arc_tag_pool
  FROM @arc_db.arc.specs SPECIFICATION_FILE = 'service.yaml'
  MIN_INSTANCES = 1 MAX_INSTANCES = 2;

CREATE OR REPLACE FUNCTION arc_db.arc.arc_tags(text VARCHAR) RETURNS VARIANT
  SERVICE = arc_db.arc.arc_tagger_svc ENDPOINT = api MAX_BATCH_ROWS = 64 AS '/v1/snowflake/tags';
CREATE OR REPLACE FUNCTION arc_db.arc.arc_tags_domain(text VARCHAR, domain VARCHAR) RETURNS VARIANT
  SERVICE = arc_db.arc.arc_tagger_svc ENDPOINT = api MAX_BATCH_ROWS = 64 AS '/v1/snowflake/tags';
-- options: '{"redact": "mask", "emit": "pii,safety"}'
CREATE OR REPLACE FUNCTION arc_db.arc.arc_tags_opts(text VARCHAR, domain VARCHAR, options VARCHAR) RETURNS VARIANT
  SERVICE = arc_db.arc.arc_tagger_svc ENDPOINT = api MAX_BATCH_ROWS = 64 AS '/v1/snowflake/tags';
