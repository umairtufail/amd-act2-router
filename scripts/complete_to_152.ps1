# Complete labeling to the current raw-task count (the script name is retained
# for compatibility with earlier handoff instructions).
#
# FASTEST: if your AMD pod already has the complete labeled dataset,
# copy it here instead of re-labeling:
#   On pod:  wc -l data/labeled_multitier.jsonl
#   Download data/labeled_multitier.jsonl + router/checkpoints/ to this folder
#
# OR label locally with Fireworks (costs tokens):
#   1. copy .env.example .env   and set FIREWORKS_API_KEY + MODEL_TIER0..3
#   2. Run this script

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$target = [int](python -c "print(sum(1 for line in open('data/tasks_raw.jsonl', encoding='utf-8') if line.strip()))")
python scripts/label_status.py --target $target

if (-not (Test-Path .env)) {
    Write-Host ""
    Write-Host "No .env file. Either:"
    Write-Host "  A) Copy data/labeled_multitier.jsonl from your AMD pod, OR"
    Write-Host "  B) copy .env.example .env and fill in Fireworks credentials"
    exit 1
}

$need = python -c "
import json
from pathlib import Path
t=$target
n=sum(1 for _ in open('data/labeled_multitier.jsonl') if _.strip()) if Path('data/labeled_multitier.jsonl').exists() else 0
print(max(t-n,0))
"

if ([int]$need -eq 0) {
    Write-Host "Already at or above $target labeled."
    exit 0
}

Write-Host "Labeling $need more tasks (batched, sleep 5s between batches)..."

while ([int](python -c "import json;from pathlib import Path;n=sum(1 for _ in open('data/labeled_multitier.jsonl') if _.strip()) if Path('data/labeled_multitier.jsonl').exists() else 0;print(n)") -lt $target) {
    python -m data.label_multitier --until-total $target --sleep 5
    python scripts/label_status.py --target $target
    Start-Sleep -Seconds 30
}

Write-Host "Done. Retrain: python -m router.train_binary_router"
