#!/usr/bin/env bash
# quant_cohort.sh — Phase A Step 0 cohort runner.
#
# Orchestrates a per-format quality bench cohort. Per variant emits:
#   - Per-tensor MSE table (quant_quality_mse, vs BF16 safetensors)
#   - KLD vs FP16 reference  [STUB — fills in once chore/113-quant-eval-plan lands]
#   - PPL on wikitext-2-test [STUB — same source]
#   - HumanEval completion capture (bench_humaneval_completion.sh)
#   - Train-pursuit reasoning smoke (default + workaround mode)
#
# Output layout matches chore/113-quant-eval-plan's results/ schema so
# kld_reduce.py can pick up the per-seq files unchanged once eval_hipfire
# starts producing them:
#
#   benchmarks/quality-baselines/results/YYYY-MM-DD-cohort-<label>/
#     ├── manifest.json                 # cohort metadata (variants, refs, flags)
#     ├── per-variant/
#     │   ├── <variant>__<arch>.mse.txt     # quant_quality_mse output
#     │   ├── <variant>__<arch>.humaneval.jsonl
#     │   ├── <variant>__<arch>.smoke.md    # train-pursuit verdict
#     │   └── <variant>__<arch>.kldseq      # ← lands here once eval_hipfire merges
#     ├── result-table.md               # aggregated markdown (built by this script)
#     └── result-table.md.in-progress   # written as we go, renamed on success
#
# Usage:
#   scripts/quant_cohort.sh <cohort-label> <bf16-ref-dir> <variant-spec.tsv>
#
# variant-spec.tsv format (tab-separated; one row per variant):
#   <variant-name>  <hfq-path>  <arch>
# Example:
#   qwen35-9b-mq4-uniform  /local/hipfire/qwen3.5-9b.mq4         gfx906
#   qwen35-9b-mfp4         /local/hipfire/qwen3.5-9b.mfp4        gfx906
#   qwen35-9b-hfp4         /local/hipfire/qwen3.5-9b.hfp4        gfx906
#
# Example invocation:
#   scripts/quant_cohort.sh \
#     phase-a-step-0-baselines \
#     /home/kread/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/SNAP \
#     /tmp/cohort_baselines.tsv

set -euo pipefail

cd "$(dirname "$0")/.."

if [ $# -lt 3 ]; then
    echo "usage: $0 <cohort-label> <bf16-ref-dir-or-file> <variant-spec.tsv>"
    echo
    echo "variant-spec.tsv format (TAB-separated):"
    echo "  <variant-name>\t<hfq-path>\t<arch>"
    exit 2
fi

LABEL="$1"
ST_DIR="$2"
SPEC="$3"

if [ ! -e "$ST_DIR" ]; then
    echo "error: bf16 reference dir/file not found: $ST_DIR"
    exit 1
fi
if [ ! -f "$SPEC" ]; then
    echo "error: variant spec not found: $SPEC"
    exit 1
fi

DATE=$(date -u +%Y-%m-%d)
COHORT_DIR="benchmarks/quality-baselines/results/${DATE}-cohort-${LABEL}"
mkdir -p "${COHORT_DIR}/per-variant"

# ─── Manifest ─────────────────────────────────────────────────────────────
HOSTNAME=$(hostname)
GIT_HEAD=$(git rev-parse HEAD 2>/dev/null || echo unknown)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)

python3 -c "
import json, os
manifest = {
    'cohort_label': '${LABEL}',
    'date_utc': '${DATE}',
    'host': '${HOSTNAME}',
    'git_head': '${GIT_HEAD}',
    'git_branch': '${GIT_BRANCH}',
    'bf16_ref': '${ST_DIR}',
    'variant_spec': '${SPEC}',
    'variants': [],
}
with open('${SPEC}') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) != 3:
            raise SystemExit(f'malformed spec row: {line!r} (expected 3 TAB-separated fields)')
        name, hfq_path, arch = parts
        manifest['variants'].append({'name': name, 'hfq_path': hfq_path, 'arch': arch})
with open('${COHORT_DIR}/manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print(f'  cohort with {len(manifest[\"variants\"])} variants')
"

# Build prerequisites once.
echo "Building prerequisites..."
cargo build --release --example quant_quality_mse --quiet 2>&1 | tail -3

PROGRESS="${COHORT_DIR}/result-table.md.in-progress"
{
    echo "# Cohort: ${LABEL}"
    echo
    echo "**Date:** ${DATE}"
    echo "**Host:** ${HOSTNAME}"
    echo "**Git:** ${GIT_BRANCH} @ ${GIT_HEAD:0:8}"
    echo "**BF16 ref:** \`${ST_DIR}\`"
    echo
    echo "## Per-variant metrics"
    echo
    echo "| Variant | Arch | MSE mean (4-bit qts) | KLD mean | KLD p99 | PPL | HE tokens (sum) | Smoke (default) | Smoke (workaround) |"
    echo "|---|---|---:|---:|---:|---:|---:|---|---|"
} > "$PROGRESS"

# Spiral-detection prompt (matches existing bench_quant_quality.sh; intentionally
# byte-identical so prior bench runs are comparable). Embedded here rather than
# read from disk to avoid newline-normalization drift.
PROMPT='A train leaves Station A traveling at 60 km/h. Two hours later, a second train leaves Station A on the same track traveling at 90 km/h. How long after the second train departs will it catch up to the first? Show your reasoning step by step.'

# ─── Per-variant loop ────────────────────────────────────────────────────
while IFS=$'\t' read -r VARIANT HFQ_PATH ARCH; do
    # Skip blank/comment rows.
    [ -z "${VARIANT:-}" ] && continue
    case "$VARIANT" in '#'*) continue ;; esac

    echo
    echo "═══ Variant: ${VARIANT} (${ARCH}) ═══"
    if [ ! -e "$HFQ_PATH" ]; then
        echo "  SKIP: model file not found: $HFQ_PATH"
        echo "| ${VARIANT} | ${ARCH} | — | — | — | — | — | MODEL_MISSING | — |" >> "$PROGRESS"
        continue
    fi

    PV="${COHORT_DIR}/per-variant/${VARIANT}__${ARCH}"

    # ─── 1. Per-tensor MSE ──────────────────────────────────────────────
    echo "  [1/4] per-tensor MSE..."
    ./target/release/examples/quant_quality_mse "$ST_DIR" "$HFQ_PATH" 2>&1 > "${PV}.mse.txt" \
        || { echo "    (MSE run failed; output captured anyway)"; }

    # Extract the aggregate 4-bit MSE (mean over qts ∈ {13, 21, 24} == MQ4/HFP4/MFP4).
    # Falls back to "—" if none of those qts are present.
    MSE_MEAN=$(python3 -c "
import re, sys
qts_4bit = {'MQ4G256', 'HFP4G32', 'MFP4G32'}
text = open('${PV}.mse.txt').read()
# parse the 'Aggregate stats by quant type' table
m = re.search(r'=== Aggregate stats by quant type ===(.*?)(?:\n\n|\Z)', text, re.S)
if not m:
    print('—'); sys.exit()
lines = [l for l in m.group(1).strip().split('\n') if l and not l.startswith('-') and not l.startswith('qt')]
mses = []
for line in lines:
    parts = line.split()
    if len(parts) >= 4 and parts[0] in qts_4bit:
        try:
            mses.append(float(parts[3]))
        except ValueError:
            pass
if not mses:
    print('—')
else:
    avg = sum(mses) / len(mses)
    print(f'{avg:.2e}')
")

    # ─── 2. KLD / PPL — STUB ────────────────────────────────────────────
    # When chore/113-quant-eval-plan merges, this section runs:
    #
    #   ./target/release/examples/eval_hipfire \
    #       --model "$HFQ_PATH" \
    #       --ref "${BF16_REF_KLDREF}" \
    #       --output "${PV}.kldseq" \
    #       --kv-mode asym3
    #
    #   then kld_reduce.py to extract slice-mean + p99 + PPL.
    #
    # Until then, emit a placeholder and a TODO comment in the per-variant dir.
    KLD_MEAN="(awaits #113)"
    KLD_P99="—"
    PPL_VAL="—"
    echo "(awaits chore/113-quant-eval-plan merge — eval_hipfire not yet in tree)" > "${PV}.kld-todo.txt"
    echo "  [2/4] KLD / PPL: STUB (awaits eval_hipfire from #113)"

    # ─── 3. HumanEval completion capture ──────────────────────────────
    echo "  [3/4] HumanEval prompts..."
    if command -v hipfire >/dev/null 2>&1; then
        if bash scripts/bench_humaneval_completion.sh "$HFQ_PATH" "${PV}.humaneval.jsonl" \
                2>&1 | tail -20 > "${PV}.humaneval.log"; then
            HE_TOK_SUM=$(python3 -c "
import json
rows = [json.loads(l) for l in open('${PV}.humaneval.jsonl')]
s = sum(r.get('tokens_used', 0) for r in rows if 'error' not in r)
print(s)
")
        else
            HE_TOK_SUM="FAIL"
            echo "    (humaneval failed; log: ${PV}.humaneval.log)"
        fi
    else
        HE_TOK_SUM="(no CLI)"
        echo "    (skip: hipfire CLI not on PATH)"
    fi

    # ─── 4. Reasoning smoke (default + workaround) ─────────────────────
    echo "  [4/4] reasoning smoke..."
    SMOKE_DEFAULT="—"
    SMOKE_WORKAROUND="—"
    if command -v hipfire >/dev/null 2>&1; then
        # Default mode
        hipfire stop 2>&1 | head -1 || true; sleep 2
        HIPFIRE_DEFAULT_MODEL="$HFQ_PATH" hipfire serve 8080 -d 2>&1 | tail -1 >/dev/null
        warmup_start=$(date +%s)
        until tail -1 ~/.hipfire/serve.log 2>/dev/null | grep -q "warm-up complete"; do
            sleep 5
            if ! pgrep -af "examples/daemon" >/dev/null; then break; fi
            if [ $(( $(date +%s) - warmup_start )) -gt 300 ]; then break; fi
        done

        MODEL_ID=$(curl -sS http://127.0.0.1:8080/v1/models 2>/dev/null \
            | python3 -c "import sys,json; ms=json.load(sys.stdin)['data']; n='$(basename "$HFQ_PATH")'; [print(m['id']) for m in ms if m['id'].endswith(n)]" \
            | head -1)
        [ -z "$MODEL_ID" ] && MODEL_ID="$(basename "$HFQ_PATH")"

        body=$(python3 -c "
import json
print(json.dumps({'model':'$MODEL_ID','messages':[{'role':'user','content':'''$PROMPT'''}],'temperature':0,'max_tokens':400}))
")
        timeout 300 curl -sS http://127.0.0.1:8080/v1/chat/completions \
            -H 'Content-Type: application/json' -d "$body" > "${PV}.smoke-default.json" 2>&1 || true

        SMOKE_DEFAULT=$(python3 -c "
import json
try:
    d = json.load(open('${PV}.smoke-default.json'))
    c = d['choices'][0]['message']['content']
    n = len(c)
    if n == 0: print('SPIRAL')
    elif n > 800: print(f'COHERENT_{n}c')
    else: print(f'PARTIAL_{n}c')
except Exception as e:
    print(f'ERR_{e}')
")
        hipfire stop 2>&1 | head -1 || true

        # Workaround mode (HIPFIRE_QWEN_MOE_FINAL_NORM_RAW=1)
        sleep 2
        HIPFIRE_QWEN_MOE_FINAL_NORM_RAW=1 HIPFIRE_DEFAULT_MODEL="$HFQ_PATH" hipfire serve 8080 -d 2>&1 | tail -1 >/dev/null
        warmup_start=$(date +%s)
        until tail -1 ~/.hipfire/serve.log 2>/dev/null | grep -q "warm-up complete"; do
            sleep 5
            if ! pgrep -af "examples/daemon" >/dev/null; then break; fi
            if [ $(( $(date +%s) - warmup_start )) -gt 300 ]; then break; fi
        done
        timeout 300 curl -sS http://127.0.0.1:8080/v1/chat/completions \
            -H 'Content-Type: application/json' -d "$body" > "${PV}.smoke-workaround.json" 2>&1 || true

        SMOKE_WORKAROUND=$(python3 -c "
import json
try:
    d = json.load(open('${PV}.smoke-workaround.json'))
    c = d['choices'][0]['message']['content']
    n = len(c)
    if n == 0: print('SPIRAL')
    elif n > 800: print(f'COHERENT_{n}c')
    else: print(f'PARTIAL_{n}c')
except Exception as e:
    print(f'ERR_{e}')
")
        hipfire stop 2>&1 | head -1 || true
    fi

    echo "| ${VARIANT} | ${ARCH} | ${MSE_MEAN} | ${KLD_MEAN} | ${KLD_P99} | ${PPL_VAL} | ${HE_TOK_SUM} | ${SMOKE_DEFAULT} | ${SMOKE_WORKAROUND} |" >> "$PROGRESS"

done < "$SPEC"

# ─── Finalize ────────────────────────────────────────────────────────────
{
    echo
    echo "## Notes"
    echo
    echo "- **MSE mean (4-bit qts):** average per-tensor MSE across MQ4G256 / HFP4G32 / MFP4G32 tensors (excluding Q8/F16-promoted tensors). Lower is better for raw reconstruction. **Not a model-quality signal alone** — see \`docs/plans/hfp4-fivetide-rebuttal-perspective.md\`."
    echo "- **KLD / PPL:** stub today. Filled in once \`chore/113-quant-eval-plan\` merges (eval_hipfire + kldref + slice corpus)."
    echo "- **HE tokens (sum):** sum of completion tokens across in-tree humaneval prompts (3 prompts). Sanity signal for 'model produces non-zero output on code prompts'; pass@1 scoring is a follow-up."
    echo "- **Smoke:** train-pursuit reasoning attractor check at temp=0, max_tokens=400. \`SPIRAL\` = empty content after \`<think>\` strip; \`COHERENT_<N>c\` = N chars of reasoning; \`PARTIAL_<N>c\` = N chars but suspiciously short."
} >> "$PROGRESS"

mv "$PROGRESS" "${COHORT_DIR}/result-table.md"

echo
echo "═══ Cohort complete ═══"
echo "  Output: ${COHORT_DIR}/result-table.md"
echo "  Per-variant artifacts: ${COHORT_DIR}/per-variant/"
echo
cat "${COHORT_DIR}/result-table.md"
