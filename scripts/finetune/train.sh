#!/usr/bin/env bash
set -euo pipefail

export NO_PROXY=100.64.0.0/10,100.126.198.114,localhost,127.0.0.1
export no_proxy=100.64.0.0/10,100.126.198.114,localhost,127.0.0.1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_CONFIG="${REPO_ROOT}/config/finetune.yaml"

if [[ $# -ge 1 && ( "${1}" == "--help" || "${1}" == "-h" ) ]]; then
  cat <<EOF
Usage: $(basename "$0") [extra python args...]

Default config: ${DEFAULT_CONFIG}
The script first formats nl2sql.jsonl into an Alpaca-style JSONL, then trains directly from that Alpaca file.

Optional environment overrides:
  FINETUNE_CONFIG
  TRL_SPATIAL_TEXT2SQL_CONFIG

Examples:
  $(basename "$0")
  $(basename "$0") --prepare-only
  $(basename "$0") --alpaca-output data/processed/finetune/custom_nl2sql_alpaca.jsonl
  $(basename "$0") --nvidia-gpu-indices 0,1,2,3,4,5,6,7
  $(basename "$0") --model-name-or-path Qwen/Qwen2.5-7B-Instruct --output-dir outputs/finetune/qwen25_7b_full
EOF
  exit 0
fi

CONFIG_PATH="${FINETUNE_CONFIG:-${TRL_SPATIAL_TEXT2SQL_CONFIG:-${DEFAULT_CONFIG}}}"

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m src.finetune.cli \
    --config "${CONFIG_PATH}" \
    "$@"
