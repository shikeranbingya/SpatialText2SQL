#!/usr/bin/env bash
set -euo pipefail

export NO_PROXY=100.64.0.0/10,100.126.198.114,localhost,127.0.0.1
export no_proxy=100.64.0.0/10,100.126.198.114,localhost,127.0.0.1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_CONFIG="${REPO_ROOT}/config/question_generation.yaml"

if [[ $# -ge 1 && ( "${1}" == "--help" || "${1}" == "-h" ) ]]; then
  cat <<EOF
Usage: $(basename "$0") [extra python args...]

Default config: ${DEFAULT_CONFIG}

Optional environment overrides:
  QUESTION_GENERATION_CONFIG

Examples:
  $(basename "$0")
  $(basename "$0") --style factual_lookup
  $(basename "$0") --sql-input data/processed/synthesized_sql_queries.jsonl --output data/processed/diversity_aware_questions.jsonl
EOF
  exit 0
fi

CONFIG_PATH="${QUESTION_GENERATION_CONFIG:-${DEFAULT_CONFIG}}"

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m src.synthesis.question.cli \
    --config "${CONFIG_PATH}" \
    "$@"
