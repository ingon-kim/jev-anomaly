#!/bin/bash
# Optional: MVTec AD test split (bottle/screw/carpet) + train/good (bottle/screw) from an HF mirror, for the eval scripts.
# License: CC BY-NC-SA 4.0 (non-commercial). Not redistributed in this repo.
set -euo pipefail
cd "$(dirname "$0")"
for c in bottle screw carpet; do
  .venv/bin/hf download TheoM55/mvtec_anomaly_detection --repo-type dataset --include "images/test/$c/**" --local-dir mvtec_raw
  mkdir -p data/$c/normal data/$c/anomaly
  for d in mvtec_raw/images/test/$c/*; do
    dn=$(basename "$d"); [ "$dn" = good ] && dst=normal || dst=anomaly
    for f in "$d"/*.png; do ln -sfr "$f" "data/$c/$dst/${dn}_$(basename "$f")"; done  # relative links survive moving the repo
  done
done
for c in bottle screw; do
  .venv/bin/hf download TheoM55/mvtec_anomaly_detection --repo-type dataset --include "images/train/$c/**" --local-dir mvtec_raw
done
