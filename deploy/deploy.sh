#!/usr/bin/env bash
# deploy.sh — pull latest code from GitHub, uv sync, restart service
# Run from project root: bash deploy/deploy.sh
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
SERVER_USER="${DEPLOY_USER:-agent}"
SERVER_HOST="${DEPLOY_HOST:?Set DEPLOY_HOST to your server IP or hostname}"
REMOTE_DIR="/home/${SERVER_USER}/agent-from-scratch"

# ── Safety: never sync user data ──────────────────────────────────────────────
# Session databases (*.db) and ChromaDB memory stores (chroma.sqlite3, *.parquet)
# contain conversation history and personal facts about users.
# They are git-ignored and MUST never travel with deploys or backups.
# If you add rsync to this script, always pass:
#   --exclude='*.db*' --exclude='chroma.sqlite3' --exclude='*.parquet'
# Full example:
#   rsync -av --exclude='*.db*' --exclude='chroma.sqlite3' \
#             --exclude='*.parquet' --exclude='.env' \
#             src/ user@host:dst/

# ── Pull latest code ──────────────────────────────────────────────────────────
echo "==> Pulling latest code on ${SERVER_USER}@${SERVER_HOST}"
ssh "${SERVER_USER}@${SERVER_HOST}" "
  cd ${REMOTE_DIR} && \
  git pull --ff-only origin master
"

# ── Install deps on server ────────────────────────────────────────────────────
echo "==> Running uv sync on server"
ssh "${SERVER_USER}@${SERVER_HOST}" "cd ${REMOTE_DIR} && uv sync --no-dev"

# ── Restart service ───────────────────────────────────────────────────────────
echo "==> Restarting agentkit.service"
ssh "${SERVER_USER}@${SERVER_HOST}" "sudo systemctl restart agentkit"

echo "==> Done. Check logs with: ssh ${SERVER_USER}@${SERVER_HOST} 'journalctl -u agentkit -f'"
