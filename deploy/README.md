# Deploying agentkit to a VPS

Target: Ubuntu 24.04, x86_64, 2 vCPU, 4 GB RAM, public IP.

---

## 1. Secure the server (one-time)

### SSH key login only

On your local PC:
```bash
ssh-keygen -t ed25519 -C "agentkit-vps"
ssh-copy-id -i ~/.ssh/id_ed25519.pub root@<SERVER_IP>
```

On the server, disable password login:
```bash
sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl reload sshd
```

### Firewall

```bash
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable
```

### fail2ban (blocks brute-force SSH)

```bash
apt install -y fail2ban
systemctl enable --now fail2ban
```

The default config protects SSH. No extra configuration needed.

---

## 2. Create a dedicated user

```bash
adduser --disabled-password --gecos "" agent
# Allow agent to restart the service without a password
echo "agent ALL=(ALL) NOPASSWD: /bin/systemctl restart agentkit, /bin/systemctl start agentkit, /bin/systemctl stop agentkit" \
  >> /etc/sudoers.d/agentkit
chmod 440 /etc/sudoers.d/agentkit
```

---

## 3. Install system dependencies

```bash
apt update && apt install -y git curl build-essential nodejs npm
```

### Install uv (as the `agent` user)

```bash
su - agent
curl -LsSf https://astral.sh/uv/install.sh | sh
# Add to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

---

## 4. Clone the repository and configure

```bash
su - agent
git clone <REPO_URL> ~/agent-from-scratch
cd ~/agent-from-scratch
```

Create `.env`:
```bash
cat > .env << 'EOF'
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
FAST_MODEL=anthropic/claude-haiku-4-5
SMART_MODEL=anthropic/claude-sonnet-5
MAX_REQUESTS_PER_HOUR=200
MAX_TOKENS_PER_HOUR=500000
LOG_LEVEL=WARNING
EOF
chmod 600 .env
```

Install dependencies (without eval group — no datasets/pandas):
```bash
uv sync --no-dev
```

### Verify setup

```bash
uv run agent doctor
```

All critical checks should show `[✓]`.

---

## 5. Install systemd service

Back as root:
```bash
cp /home/agent/agent-from-scratch/deploy/agentkit.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now agentkit
systemctl status agentkit
```

---

## 6. Deploy updates (from your local PC)

Set server details once:
```bash
export DEPLOY_HOST=<SERVER_IP>
export DEPLOY_USER=agent   # default, can omit
```

Then deploy:
```bash
bash deploy/deploy.sh
```

The script:
1. `rsync` — copies code, skips `.env`, `.venv`, `results/`, `.git`
2. `uv sync --no-dev` — installs/updates dependencies on server
3. `systemctl restart agentkit` — restarts the service

---

## 7. View logs

```bash
# Follow live logs
journalctl -u agentkit -f

# Last 100 lines
journalctl -u agentkit -n 100

# Logs since last boot
journalctl -u agentkit -b
```

---

## 8. Run the benchmark (compare PC vs server)

On your PC:
```bash
uv run python scripts/bench.py
# saves results/bench-<hostname>.json
```

On the server:
```bash
uv run python scripts/bench.py
# saves results/bench-<hostname>.json
```

Copy server result to PC and compare:
```bash
scp agent@<SERVER>:~/agent-from-scratch/results/bench-*.json results/
uv run python scripts/bench.py --compare results/bench-<pc>.json results/bench-<server>.json
```
