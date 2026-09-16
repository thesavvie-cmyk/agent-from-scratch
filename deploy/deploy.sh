#!/usr/bin/env bash
# deploy.sh — sync code from local PC to server, uv sync, restart service
# Run from project root: bash deploy/deploy.sh
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
SERVER_USER="${DEPLOY_USER:-agent}"
SERVER_HOST="${DEPLOY_HOST:?Set DEPLOY_HOST to your server IP or hostname}"
REMOTE_DIR="/home/${SERVER_USER}/agent-from-scratch"

# ── Sync ──────────────────────────────────────────────────────────────────────
echo "==> Syncing code to ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}"
rsync -az --delete \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='.env' \
  --exclude='results/' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  --exclude='*.egg-info' \
  . "${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/"

# ── Install deps on server ────────────────────────────────────────────────────
echo "==> Running uv sync on server"
ssh "${SERVER_USER}@${SERVER_HOST}" "cd ${REMOTE_DIR} && uv sync --no-dev"

# ── Restart service ───────────────────────────────────────────────────────────
echo "==> Restarting agentkit.service"
ssh "${SERVER_USER}@${SERVER_HOST}" "sudo systemctl restart agentkit"

echo "==> Done. Check logs with: ssh ${SERVER_USER}@${SERVER_HOST} 'journalctl -u agentkit -f'"
