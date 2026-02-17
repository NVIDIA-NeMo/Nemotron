#!/usr/bin/env bash
# Download scripts from the bird_sql directory on GitHub:
#   haydeniw/Nemotron, branch hwolff/megatron-bridge-launchable
#   path: usage-cookbook/Nemotron-3-Nano/finetuning_and_deployment/bird_sql
# Saves files into ~/bird_sql (home directory).

set -e

REPO="haydeniw/Nemotron"
BRANCH="hwolff/megatron-bridge-launchable"
REMOTE_PATH="usage-cookbook/Nemotron-3-Nano/finetuning_and_deployment/bird_sql"
API_URL="https://api.github.com/repos/${REPO}/contents/${REMOTE_PATH}?ref=${BRANCH}"

OUT_DIR="${HOME}/bird_sql"
mkdir -p "$OUT_DIR"

if ! command -v jq &>/dev/null; then
  echo "jq is required. Install with: brew install jq"
  exit 1
fi

curl -sL -H "Accept: application/vnd.github.v3+json" "$API_URL" \
  | jq -r '.[] | select(.type == "file") | "\(.download_url)\t\(.name)"' \
  | while IFS=$'\t' read -r url name; do
  echo "Downloading $name -> $OUT_DIR/$name"
  curl -sL "$url" -o "$OUT_DIR/$name"
done

echo "Done. Files in $OUT_DIR"
