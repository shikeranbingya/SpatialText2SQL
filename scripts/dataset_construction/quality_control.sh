#!/usr/bin/env bash
set -euo pipefail

export NO_PROXY=100.64.0.0/10,100.126.198.114,localhost,127.0.0.1
export no_proxy=100.64.0.0/10,100.126.198.114,localhost,127.0.0.1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_CONFIG="${REPO_ROOT}/config/quality_control.yaml"

if [[ $# -ge 1 && ( "${1}" == "--help" || "${1}" == "-h" ) ]]; then
  cat <<EOF
Usage: $(basename "$0") [extra python args...]

Default config: ${DEFAULT_CONFIG}

Optional environment overrides:
  QUALITY_CONTROL_CONFIG

Examples:
  $(basename "$0")
  $(basename "$0") --semantic-mode warning_only
  $(basename "$0") --input data/processed/diversity_aware_questions.jsonl --output data/processed/quality_controlled_nl_sql.jsonl
EOF
  exit 0
fi

CONFIG_PATH="${QUALITY_CONTROL_CONFIG:-${DEFAULT_CONFIG}}"

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m src.synthesis.quality.cli \
    --config "${CONFIG_PATH}" \
    "$@"
