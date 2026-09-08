#!/usr/bin/env bash
# Fetch the Numenta Anomaly Benchmark (NAB) data corpus and anomaly-window
# labels into data/raw/. Only benchmark artefacts are copied; no NAB source
# code is downloaded or used. data/raw/ is gitignored, so run this once after
# cloning.
#
# Requires: git
set -euo pipefail

NAB_REPO="https://github.com/numenta/NAB.git"
NAB_COMMIT="ea702d75cc2258d9d7dd35ca8e5e2539d71f3140"  # chore: MIT License (2024-12-03)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/data/raw"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

echo "Cloning NAB (${NAB_COMMIT:0:12}) into a temporary directory..."
git clone --quiet --filter=blob:none --no-checkout --sparse "${NAB_REPO}" "${TMP}/NAB"
cd "${TMP}/NAB"
git sparse-checkout set data labels
git checkout --quiet "${NAB_COMMIT}"

echo "Copying data corpus and labels into ${DEST} ..."
mkdir -p "${DEST}/labels"
rm -rf "${DEST}/data"
cp -r "${TMP}/NAB/data" "${DEST}/data"
cp "${TMP}/NAB/labels/combined_windows.json" "${DEST}/labels/combined_windows.json"
cp "${TMP}/NAB/labels/combined_labels.json" "${DEST}/labels/combined_labels.json"

csv_count="$(find "${DEST}/data" -name '*.csv' | wc -l | tr -d ' ')"
echo "Done. ${csv_count} CSV series under data/raw/data/, labels under data/raw/labels/."
