#!/usr/bin/env bash
# Sync QuantAb code to EC2 and optionally launch experiments in tmux.
#
# Usage:
#   bash scripts/run_on_ec2.sh            # sync + run two-register experiments
#   bash scripts/run_on_ec2.sh --sync-only
#   bash scripts/run_on_ec2.sh --shell    # sync and drop into interactive shell
#
# Requires (not stored in the repo):
#   export QUANTAB_INSTANCE_ID=<instance-id> AWS_PROFILE=<your-profile> QUANTAB_S3_BUCKET=<your-bucket>
#
# The EICE key push window is 60 s — the script handles this automatically.

set -euo pipefail

INSTANCE_ID="${QUANTAB_INSTANCE_ID:?export QUANTAB_INSTANCE_ID (EC2 instance ID)}"
AWS_PROFILE="${AWS_PROFILE:?export AWS_PROFILE (your AWS SSO profile)}"
S3_BUCKET="${QUANTAB_S3_BUCKET:?export QUANTAB_S3_BUCKET (team bucket name)}"
AWS_REGION="us-west-2"
REMOTE_USER="ec2-user"
KEY_PATH="/tmp/eice-quantab-key"
REMOTE_DIR="/home/ec2-user/quantab"
SESSION_NAME="quantab"

SYNC_ONLY=false
DROP_SHELL=false
for arg in "$@"; do
  case $arg in
    --sync-only) SYNC_ONLY=true ;;
    --shell)     DROP_SHELL=true ;;
  esac
done

# ── 1. Generate a fresh EICE key ──────────────────────────────────────────────
echo "==> Generating EICE key …"
ssh-keygen -t ed25519 -N "" -f "$KEY_PATH" -q -y 2>/dev/null || \
  ssh-keygen -t ed25519 -N "" -f "$KEY_PATH" -q

# ── 2. Push key to instance (valid for 60 s) ──────────────────────────────────
echo "==> Pushing public key to instance …"
aws ec2-instance-connect send-ssh-public-key \
  --instance-id "$INSTANCE_ID" \
  --instance-os-user "$REMOTE_USER" \
  --ssh-public-key "file://${KEY_PATH}.pub" \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION"

# Common SSH options (tunnel through EICE, 55 s timeout before key expires)
SSH_OPTS=(
  -i "$KEY_PATH"
  -o StrictHostKeyChecking=no
  -o ConnectTimeout=30
  -o ProxyCommand="aws ec2-instance-connect open-tunnel \
      --instance-id $INSTANCE_ID \
      --profile $AWS_PROFILE --region $AWS_REGION"
)

# ── 3. Sync code ──────────────────────────────────────────────────────────────
echo "==> Syncing code to ${REMOTE_USER}@${INSTANCE_ID}:${REMOTE_DIR} …"
# Use rsync over the SSH tunnel; exclude heavy artefacts
rsync -az --delete \
  -e "ssh ${SSH_OPTS[*]}" \
  --exclude "*.pyc" \
  --exclude "__pycache__" \
  --exclude ".git" \
  --exclude "data/" \
  --exclude "embeddings/" \
  --exclude "results/" \
  --exclude "*.tar.gz" \
  . "${REMOTE_USER}@${INSTANCE_ID}:${REMOTE_DIR}/"

echo "==> Code synced."

[[ "$SYNC_ONLY" == true ]] && echo "Done (sync-only)." && exit 0

# ── 4. Bootstrap environment on first run ─────────────────────────────────────
echo "==> Bootstrapping Python environment …"
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${INSTANCE_ID}" bash <<'REMOTE'
set -euo pipefail
cd ~/quantab

# Install into user site if venv not present
if [ ! -d venv ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -e . 2>&1 | tail -5
echo "Environment ready."
REMOTE

# ── 5. Sync embeddings from S3 (skip if already present) ─────────────────────
echo "==> Syncing embeddings from S3 …"
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${INSTANCE_ID}" "S3_BUCKET='${S3_BUCKET}' bash" <<'REMOTE'
set -euo pipefail
cd ~/quantab
source venv/bin/activate
mkdir -p embeddings
# Only download PCA-reduced files (skip raw arrays, ~160 MB each)
aws s3 sync "s3://${S3_BUCKET}/embeddings/" embeddings/ \
  --region us-west-2 \
  --exclude "*_raw.npy" \
  --no-progress \
  || echo "[warn] S3 sync failed — using whatever is already in embeddings/"
echo "Embeddings ready."
REMOTE

# ── 6. Drop into interactive shell or launch experiments ──────────────────────
if [[ "$DROP_SHELL" == true ]]; then
  echo "==> Opening interactive shell …"
  ssh "${SSH_OPTS[@]}" -t "${REMOTE_USER}@${INSTANCE_ID}" \
    "cd ~/quantab && source venv/bin/activate && exec bash"
  exit 0
fi

echo "==> Launching experiments in tmux session '${SESSION_NAME}' …"
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${INSTANCE_ID}" bash <<REMOTE
set -euo pipefail
cd ~/quantab
source venv/bin/activate

# Kill any previous session with the same name
tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true

tmux new-session -d -s "${SESSION_NAME}" -x 220 -y 50

# Window 0: two-register experiments (primary run)
tmux send-keys -t "${SESSION_NAME}:0" \
  "cd ~/quantab && source venv/bin/activate && \\
   python scripts/run_two_register.py \\
     --n-components 6 10 \\
     --train-sizes 50 100 200 500 1000 2000 \\
     --quantum-max-train 2000 \\
     --vqc-max-train 100 \\
     --vqc-epochs 50 \\
     2>&1 | tee results/two_reg_run.log" Enter

# Window 1: classical-only quick run (for early preview)
tmux new-window -t "${SESSION_NAME}"
tmux send-keys -t "${SESSION_NAME}:1" \
  "cd ~/quantab && source venv/bin/activate && \\
   python scripts/run_two_register.py \\
     --n-components 6 10 \\
     --train-sizes 50 100 200 500 1000 2000 5000 \\
     --no-quantum --no-vqc \\
     --overwrite \\
     2>&1 | tee results/classical_run.log" Enter

echo ""
echo "Experiments launched in tmux session '${SESSION_NAME}'."
echo "Attach with:  tmux attach -t ${SESSION_NAME}"
REMOTE

echo ""
echo "==> Done. To watch progress, SSH in and run:"
echo "    tmux attach -t ${SESSION_NAME}"
echo ""
echo "SSH command (re-run within 60 s of key push):"
echo "  ssh -i $KEY_PATH -o StrictHostKeyChecking=no \\"
echo "    -o ProxyCommand='aws ec2-instance-connect open-tunnel \\"
echo "      --instance-id $INSTANCE_ID --profile $AWS_PROFILE --region $AWS_REGION' \\"
echo "    ${REMOTE_USER}@${INSTANCE_ID}"
