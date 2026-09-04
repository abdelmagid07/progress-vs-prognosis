#!/usr/bin/env bash
# Real SWE-bench batch with mini-swe-agent driving on-policy Qwen3.
#
# This is the generation path for the agent trajectories the paper analyses.
#
# Topology:
#   - Local host: mini-swe-agent + Docker containers + tests.
#   - Model inference: self-hosted vLLM *or* Azure Foundry managed compute.
#     Switch with MODEL_PROVIDER (see configs/providers/).
#
# Prerequisites (local):
#   1. Docker daemon running and reachable.
#   2. pip install -e ".[agent]" (installs mini-swe-agent + the swebench harness)
#   3. A reachable remote model endpoint (set MODEL_API_BASE).
#
# Environment:
#   MODEL_PROVIDER   vllm (default) | azure
#                    Selects configs/providers/<name>.yaml (request-shape adapter).
#   MODEL_API_BASE   OpenAI-compatible base URL
#                    vllm:  http://localhost:8000/v1  or tunnel .../v1
#                    azure: https://<resource>.services.ai.azure.com/openai/v1
#   MODEL_API_KEY    API key (default: EMPTY for local vLLM)
#   MODEL_NAME       litellm model id
#                    vllm:  hosted_vllm/Qwen3-32B (the model the paper reports)
#                    azure: openai/<deployment>   e.g. openai/qwen3-32b
#   SUBSET           SWE-bench subset: verified | lite | <dataset path> (default: verified)
#   SPLIT            dataset split (default: test)
#   WORKERS          parallel workers (default: 1)
#   STEP_LIMIT       agent.step_limit (default: 60, the reported step budget)
#   ROLLOUTS         seed-only rollouts per task (default: 1, as reported)
#   SEED_BASE        first sampling seed; rollouts use SEED_BASE..SEED_BASE+ROLLOUTS-1 (default: 0)
#   OUTPUT_DIR       default data/trajectories/mini_swe_run_<timestamp>
#   SKIP_PREFLIGHT   set to 1 to skip the endpoint connectivity check
#
# Multi-rollout layout: with ROLLOUTS>1 each rollout is generated with a distinct
# sampling seed (injected as extra_body.seed) and written to OUTPUT_DIR/r<seed>/,
# so the same task appears once per seed. ingest_batch stamps task_id/seed from
# this layout. With ROLLOUTS=1 the run stays flat in OUTPUT_DIR (seed unset).
#
# Usage:
#   # vLLM 8B (default)
#   export MODEL_API_BASE="http://localhost:8000/v1"
#   bash scripts/run_mini_swe_batch.sh
#
#   # Azure Foundry 32B
#   export MODEL_PROVIDER=azure
#   export MODEL_API_BASE="https://<resource>.services.ai.azure.com/openai/v1"
#   export MODEL_API_KEY="..."
#   export MODEL_NAME="openai/qwen3-32b"
#   bash scripts/run_mini_swe_batch.sh
#
# Any extra args after the instances file are forwarded to `mini-extra swebench`.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PROVIDER="${MODEL_PROVIDER:-vllm}"
MODEL_API_BASE="${MODEL_API_BASE:-${VLLM_URL:-http://localhost:8000/v1}}"
MODEL_API_KEY="${MODEL_API_KEY:-EMPTY}"
if [[ "$MODEL_PROVIDER" == "azure" ]]; then
  MODEL_NAME="${MODEL_NAME:-openai/qwen3-32b}"
else
  MODEL_NAME="${MODEL_NAME:-hosted_vllm/Qwen3-32B}"
fi
SUBSET="${SUBSET:-verified}"
SPLIT="${SPLIT:-test}"
WORKERS="${WORKERS:-1}"
STEP_LIMIT="${STEP_LIMIT:-60}"
ROLLOUTS="${ROLLOUTS:-1}"
# Track whether the caller pinned a seed (regens do: SEED_BASE=<fresh> ROLLOUTS=1).
if [[ -n "${SEED_BASE+x}" ]]; then SEED_BASE_EXPLICIT=1; else SEED_BASE_EXPLICIT=0; fi
SEED_BASE="${SEED_BASE:-0}"
# Nest each rollout under r<seed>/ and inject its seed when doing multiple
# rollouts or when a specific seed was pinned; a plain single run stays flat.
if [[ "$ROLLOUTS" -gt 1 || "$SEED_BASE_EXPLICIT" == "1" ]]; then NEST_SEED=1; else NEST_SEED=0; fi
# Optional: a file of SWE-bench instance ids, one per line, to restrict the run.
# With no argument the full $SUBSET/$SPLIT subset runs (Verified is the 500-task set).
INSTANCES_FILE="${1:-}"
shift || true
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-data/trajectories/mini_swe_run_${TIMESTAMP}}"
OVERRIDE_CFG="configs/mini_swe_qwen.yaml"
PROVIDER_CFG="configs/providers/${MODEL_PROVIDER}.yaml"

if [[ ! -f "$PROVIDER_CFG" ]]; then
  echo "ERROR: unknown MODEL_PROVIDER=$MODEL_PROVIDER (no $PROVIDER_CFG)"
  echo "  Known: vllm | azure"
  exit 1
fi

if ! command -v mini-extra >/dev/null 2>&1; then
  echo "ERROR: mini-extra not found. Install with: pip install -e \".[agent]\""
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker is not reachable. Start the Docker daemon."
  exit 1
fi

# Fail fast if the remote model endpoint is unreachable, before spinning up Docker.
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  MODELS_URL="${MODEL_API_BASE%/}/models"
  echo "Preflight: checking $MODELS_URL ..."
  echo "  provider=$MODEL_PROVIDER  model=$MODEL_NAME  key_len=${#MODEL_API_KEY}"
  PREFLIGHT_OK=0
  PREFLIGHT_ERR=""
  # Azure Foundry accepts either Bearer or api-key; try Bearer first (OpenAI SDK style).
  for AUTH_HDR in \
      "Authorization: Bearer ${MODEL_API_KEY}" \
      "api-key: ${MODEL_API_KEY}"; do
    PREFLIGHT_ERR="$(curl -sS -m 20 -w "\nHTTP %{http_code}" \
      -H "$AUTH_HDR" "$MODELS_URL" 2>&1)" || true
    CODE="$(printf '%s\n' "$PREFLIGHT_ERR" | sed -n 's/^HTTP //p' | tail -1)"
    if [[ "$CODE" == "200" ]]; then
      PREFLIGHT_OK=1
      echo "Preflight: endpoint reachable (HTTP 200 via ${AUTH_HDR%%:*})."
      break
    fi
  done
  if [[ "$PREFLIGHT_OK" != "1" ]]; then
    echo "ERROR: cannot reach $MODELS_URL"
    echo "Last response (truncated):"
    printf '%s\n' "$PREFLIGHT_ERR" | head -c 500; echo
    echo ""
    echo "Checklist:"
    echo "  - MODEL_API_BASE must be .../openai/v1  (not .../chat/completions,"
    echo "    and not .../managed-deployments/... for this preflight)"
    echo "  - Export vars in THIS shell:  set -a && source .env && set +a"
    echo "  - Azure deploy still Succeeded + Playground chat works?"
    echo "  - Temporary bypass:  SKIP_PREFLIGHT=1 bash scripts/run_mini_swe_batch.sh ..."
    exit 1
  fi
fi

mkdir -p "$OUTPUT_DIR"

# Build a --filter regex from an instance-id file, if one is present. Built once;
# reused across every rollout so all seeds cover the same task set.
FILTER_ARGS=()
if [[ -f "$INSTANCES_FILE" ]]; then
  FILTER="$(grep -vE '^\s*(#|$)' "$INSTANCES_FILE" | paste -sd'|' -)"
  if [[ -n "$FILTER" ]]; then
    FILTER_ARGS=(--filter "$FILTER")
    echo "Filtering to $(grep -cvE '^\s*(#|$)' "$INSTANCES_FILE") instance ids from $INSTANCES_FILE"
  fi
else
  echo "No instances file given; running the full $SUBSET/$SPLIT subset."
fi

echo "=== mini-swe-agent SWE-bench batch (local Docker -> remote model) ==="
echo "Provider:       $MODEL_PROVIDER  ($PROVIDER_CFG)"
echo "Model endpoint: $MODEL_API_BASE"
echo "Model name:     $MODEL_NAME"
echo "Subset/split:   $SUBSET / $SPLIT   workers=$WORKERS   step_limit=$STEP_LIMIT"
echo "Rollouts:       $ROLLOUTS (seeds ${SEED_BASE}..$((SEED_BASE + ROLLOUTS - 1)))"
echo "Output dir:     $OUTPUT_DIR"
echo ""

# One generation pass per rollout seed. Each pass re-resolves the config so the
# sampling seed is baked into extra_body.seed, and writes to its own subdir when
# doing more than one rollout (so ingest can recover the seed from the layout).
LAST_SEED=$((SEED_BASE + ROLLOUTS - 1))
for SEED in $(seq "$SEED_BASE" "$LAST_SEED"); do
  if [[ "$NEST_SEED" == "1" ]]; then
    RUN_DIR="$OUTPUT_DIR/r${SEED}"
    INJECT_SEED="$SEED"
  else
    RUN_DIR="$OUTPUT_DIR"
    INJECT_SEED=""   # plain single run stays flat / seed-agnostic
  fi
  mkdir -p "$RUN_DIR"
  RESOLVED_CONFIG="$RUN_DIR/mini_swe_resolved.yaml"

  # Resolve: installed base <- shared override <- provider <- env.
  MODEL_API_BASE="$MODEL_API_BASE" MODEL_API_KEY="$MODEL_API_KEY" \
  MODEL_NAME="$MODEL_NAME" STEP_LIMIT="$STEP_LIMIT" SEED="$INJECT_SEED" \
  OVERRIDE_CFG="$OVERRIDE_CFG" PROVIDER_CFG="$PROVIDER_CFG" \
  RESOLVED_CONFIG="$RESOLVED_CONFIG" \
  python scripts/resolve_mini_config.py

  echo ""
  echo "--- rollout seed=$SEED -> $RUN_DIR ---"
  mini-extra swebench \
    -c "$RESOLVED_CONFIG" \
    -m "$MODEL_NAME" \
    --subset "$SUBSET" \
    --split "$SPLIT" \
    -w "$WORKERS" \
    -o "$RUN_DIR" \
    "${FILTER_ARGS[@]}" \
    "$@"
done

echo ""
echo "=== Done ==="
echo "Trajectories + predictions in: $OUTPUT_DIR (per-instance *.traj.json + preds.json"
echo "under r<seed>/ when ROLLOUTS>1)."
echo ""
echo "Next — evaluate EACH rollout's predictions (outcomes are per-rollout), place"
echo "each report as results.json next to that rollout's preds.json, then ingest:"
if [[ "$NEST_SEED" == "1" ]]; then
  echo "  for d in $OUTPUT_DIR/r*/; do"
  echo "    python -m swebench.harness.run_evaluation \\"
  echo "      --dataset_name princeton-nlp/SWE-bench_Verified \\"
  echo "      --predictions_path \"\$d/preds.json\" --run_id \"\$(basename \$d)\""
  echo "    # place the harness report at \$d/results.json"
  echo "  done"
else
  echo "  python -m swebench.harness.run_evaluation \\"
  echo "    --dataset_name princeton-nlp/SWE-bench_Verified \\"
  echo "    --predictions_path $OUTPUT_DIR/preds.json --run_id <run_id>"
  echo "  # place the harness report at $OUTPUT_DIR/results.json"
fi
echo "  # then (ingest walks r<seed>/ subdirs and reads each rollout's results.json):"
echo "  python -m prognosis.agent.ingest_batch --traj-dir $OUTPUT_DIR --format mini-swe-agent"
echo ""
echo "Then run the GPU projection step (prognosis.extract.project_steps) on a GPU host,"
echo "not locally — it needs raw residual-stream activations, which the API cannot give."
