#!/usr/bin/env bash
set -euo pipefail

export NO_PROXY=100.64.0.0/10,100.126.198.114,localhost,127.0.0.1
export no_proxy=100.64.0.0/10,100.126.198.114,localhost,127.0.0.1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_CONFIG="${REPO_ROOT}/config/sql_synthesis.yaml"

if [[ $# -ge 1 && ( "${1}" == "--help" || "${1}" == "-h" ) ]]; then
  cat <<EOF
Usage: $(basename "$0") [extra python args...]

Default config: ${DEFAULT_CONFIG}

Optional environment overrides:
  SQL_SYNTHESIS_CONFIG

Examples:
  $(basename "$0")
  $(basename "$0") --difficulty hard --num-sql-per-database 3
  $(basename "$0") --input data/processed/synthesized_spatial_databases.jsonl --output data/processed/synthesized_sql_queries.jsonl
EOF
  exit 0
fi

CONFIG_PATH="${SQL_SYNTHESIS_CONFIG:-${DEFAULT_CONFIG}}"

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m src.synthesis.sql.cli \
    --config "${CONFIG_PATH}" \
    "$@"
