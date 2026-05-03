# SpatialText2SQL Crawler

The unified entry point for the crawler in this repository is:

```bash
scripts/dataset_construction/crawl_open_data_maps.sh
```

By default, it will download map datasets from all 7 cities. The format is unified as GeoJSON, and the data is saved to:

```text
data/raw/<city_name>/geojson/
```

The current city directory names are: `new_york_city`, `los_angeles`, `chicago`, `seattle`, `san_francisco`, `boston`, and `phoenix`.

The crawler will prioritize reading the existing `data/raw/metadata.json`:

- If a dataset already appears in `metadata.json`, it will not be downloaded again by default.
- The default mode is "append/skip", which will not overwrite existing data.
- The `metadata.json` will only be written back after the datasets for all cities have been processed.

A single summarized metadata file will be generated:

```text
data/raw/metadata.json
```

`metadata.json` is a JSON array, where each object corresponds to a city and contains statistical fields such as `City`, `#Table`, `#Field/Table`, `#Spatial Field/Table`, `#Row/Table`, etc.

## Common Usage

Download all map data for all cities (Full Download):

```bash
scripts/dataset_construction/crawl_open_data_maps.sh
```

Download a sample of 10 datasets for each city:

```bash
scripts/dataset_construction/crawl_open_data_maps.sh 10
```

*Note: You can also use other parameters to customize the behavior, which are detailed below.*

## Key Parameters

- `--sample N`: Download at most `N` datasets per city. If omitted, downloads all map data for all cities.
- `--cities LIST`: Comma-separated list of cities. Options: `nyc,lacity,chicago,seattle,sf,boston,phoenix`. Default: `all`.
- `--out-root PATH`: Root directory for downloads. Default: `data/raw`.
- `--metadata-name NAME`: Filename for the root metadata. Default: `metadata.json`.
- `--page-size N`: Pagination size for catalog APIs. Default: `100`.
- `--row-limit N`: Maximum number of rows exported by Socrata GeoJSON fallback. Default: `5000000`.
- `--sleep SECONDS`: Waiting time between two downloads. Default: `0`.
- `--timeout SECONDS`: HTTP request timeout. Default: `120`.
- `--override`: Force overwrite existing datasets. Default is not to overwrite (skips datasets already present in `metadata.json`).
- `--list-cities`: Print configured city ids and exit.

If the volume of requests to Socrata is large, you can configure `SOCRATA_APP_TOKEN`:

```bash
SOCRATA_APP_TOKEN=your_token scripts/dataset_construction/crawl_open_data_maps.sh --sample 20
```

## Table Canonicalization

Canonicalize the crawled metadata and enrich each dataset entry in place:

```bash
scripts/dataset_construction/table_canonicalization.sh data/raw/metadata.json
```

By default this writes:

```text
data/raw/metadata_canonicalized.json
```

The canonicalization step updates each dataset in place:

- dataset-level `canonical_name`
- column-level `canonical_name` and `canonical_type`
- dataset-level `spatial_fields`

`nullable` is not written into the canonicalized metadata.

City selection matches `crawl_open_data_maps.sh`. For example:

```bash
scripts/dataset_construction/table_canonicalization.sh data/raw/metadata.json --cities nyc,sf
```

## Spatial Database Synthesis

The relation-aware synthesis entrypoint is:

```bash
scripts/dataset_construction/synthesize_spatial_databases.sh
```

The shell wrapper calls the Python CLI in:

```text
src/synthesis/database/cli.py
```

Default behavior:

- Input: `data/raw/metadata_canonicalized.json`
- Output: `data/processed/synthesized_spatial_databases.jsonl`
- Number of synthesized databases per city: automatically set to `ceil(num_tables_in_city / 10)`
- `TARGET_AVG_DEGREE=4`
- `EXPLORATION_PROB=0.1`
- `SIZE_MEAN=8`
- `SIZE_STD=2`
- `MIN_TABLES=2`
- `MAX_TABLES=12`
- `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `RANDOM_SEED=42`
- `MAX_SAMPLING_STEPS=100`
- `LOG_LEVEL=INFO`

Run with all defaults:

```bash
scripts/dataset_construction/synthesize_spatial_databases.sh
```

Override the input and output with positional arguments:

```bash
scripts/dataset_construction/synthesize_spatial_databases.sh \
  data/raw/metadata_canonicalized.json \
  data/processed/synthesized_spatial_databases.jsonl
```

Restrict synthesis to selected cities using the same `--cities` syntax as `crawl_open_data_maps.sh`:

```bash
scripts/dataset_construction/synthesize_spatial_databases.sh \
  data/raw/metadata_canonicalized.json \
  data/processed/synthesized_spatial_databases.jsonl \
  --cities nyc,sf
```

Override other sampling defaults through environment variables, and pass any extra CLI flags after the first two positional arguments:

```bash
EXPLORATION_PROB=0.2 \
scripts/dataset_construction/synthesize_spatial_databases.sh \
  data/raw/metadata_canonicalized.json \
  data/processed/synthesized_spatial_databases.jsonl \
  --cities nyc,sf \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

This stage requires `networkx`, and the default embedding backend also requires `sentence-transformers`.

## PostGIS Migration

Migrate synthesized spatial databases into PostGIS schemas:

```bash
scripts/dataset_construction/migrate_synthesized_spatial_databases.sh
```

Common options:

```bash
# Override input file
scripts/dataset_construction/migrate_synthesized_spatial_databases.sh \
  data/processed/synthesized_spatial_databases.jsonl

# Restrict to specific cities
scripts/dataset_construction/migrate_synthesized_spatial_databases.sh \
  data/processed/synthesized_spatial_databases.jsonl \
  --cities nyc,sf

# Control bulk insert batch size
INSERT_BATCH_SIZE=2000 \
scripts/dataset_construction/migrate_synthesized_spatial_databases.sh

# Truncate loaded source rows per table (-1 disables truncation)
SOURCE_ROW_LIMIT=500000 \
scripts/dataset_construction/migrate_synthesized_spatial_databases.sh
```

Database configuration defaults:

- Host: `localhost`
- Port: `5432`
- User: `postgres`
- Password: `123456`
- Catalog: `syntheized`
- Bootstrap database: `postgres`
- Insert batch size: `10000`
- Source row limit per table: `500000` (`-1` disables truncation)

`bootstrap_db` is only the bootstrap connection used to check or create the shared target catalog. The synthesized databases themselves are migrated as schemas inside the target catalog.

Migration behavior:

- Each synthesized `database_id` becomes one schema inside the shared catalog.
- The migrator recreates that schema and then imports the selected tables into it.
- GeoJSON features are inserted in batches, not one row at a time.

Edit persistent settings in `config/migrate.yaml`.

## SQL Synthesis

Generate PostGIS SQL samples from synthesized spatial databases:

```bash
scripts/dataset_construction/synthesize_sql_queries.sh
```

Common options:

```bash
# Override input, output, and difficulty
scripts/dataset_construction/synthesize_sql_queries.sh \
  --input data/processed/synthesized_spatial_databases.jsonl \
  --output data/processed/synthesized_sql_queries.jsonl \
  --difficulty hard \
  --num-sql-per-database 3

# Dry run without execution checks
scripts/dataset_construction/synthesize_sql_queries.sh \
  --disable-execution-check \
  --dry-run
```

Default settings:

- Input: `data/processed/synthesized_spatial_databases.jsonl`
- Output: `data/processed/synthesized_sql_queries.jsonl`
- Database connection: `localhost:5432/syntheized`
- `num_sql_per_database` supports per-city mapping, for example `nyc=8,sf=6` from CLI or a YAML mapping in `config/sql_synthesis.yaml`
- The default config only enables `nyc: 8`; cities not listed will not emit SQL unless you add them or provide a `default` entry

Edit persistent settings in `config/sql_synthesis.yaml`.

## PostGIS Docs Parse

Use the unified entry point below for PostGIS documentation parsing workflows:

```bash
scripts/postgis_docs_parse/run_postgis_docs_parse.sh
```

Common commands:

```bash
scripts/postgis_docs_parse/run_postgis_docs_parse.sh extract --input-dir xml_data --output-file extract_result/postgis_extracted.json
scripts/postgis_docs_parse/run_postgis_docs_parse.sh validate --input extract_result/postgis_extracted.json --output validation_result/postgis_validated.json --review manual_review/manual_review.json
```

## Benchmarks

Benchmark implementations now live under:

```text
src/benchmark/<benchmark_name>/
```

Use the shell entrypoints under `scripts/benchmark/` to run them.

### FloodSQL

Typical commands:

```bash
scripts/benchmark/floodsql/migrate_to_postgis.sh
scripts/benchmark/floodsql/validate_gold_sql.sh --utils-first
scripts/benchmark/floodsql/build_execution_consistency.sh
```

Reports are written to `scripts/benchmark/floodsql/`.

### Spatial QA

Create or inspect the PostgreSQL indexes used by the benchmark:

```bash
scripts/benchmark/spatial_qa/create_benchmark_indexes.sh
scripts/benchmark/spatial_qa/create_benchmark_indexes.sh --check-only
```

### SpatialSQL

Fetch the dataset, validate the integration, then run the migration workflow:

```bash
scripts/benchmark/spatialsql/fetch_sdbdatasets.sh
scripts/benchmark/spatialsql/verify_adaptation.sh
scripts/benchmark/spatialsql/migrate_to_separate_db.sh
scripts/benchmark/spatialsql/validate_gold_sql.sh --utils-first
```

Legacy schema-per-database migration is still available at:

```bash
scripts/benchmark/spatialsql/migrate_sqlite_to_pg.sh
```
