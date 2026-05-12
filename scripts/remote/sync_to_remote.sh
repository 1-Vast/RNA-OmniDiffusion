#!/bin/bash
# Deploy local clean mainline to remote 4090 server
# Run from LOCAL (Windows WSL or Git Bash):
#   bash scripts/remote/sync_to_remote.sh user@host

REMOTE="${1:?Usage: $0 user@remote_host}"
REMOTE_PATH="/root/autodl-tmp/RNA-OmniDiffusion"

echo "=== Syncing to $REMOTE ==="

# Backup remote dataset and .env first
ssh "$REMOTE" "
    cd /root/autodl-tmp
    if [ -d RNA-OmniDiffusion/dataset ]; then
        mkdir -p _backup
        cp -r RNA-OmniDiffusion/dataset _backup/dataset 2>/dev/null || true
        cp RNA-OmniDiffusion/.env _backup/.env 2>/dev/null || true
        echo 'Backup created'
    fi
"

# Clean remote (keep dataset + .env)
ssh "$REMOTE" "
    cd /root/autodl-tmp/RNA-OmniDiffusion 2>/dev/null || mkdir -p /root/autodl-tmp/RNA-OmniDiffusion
    cd /root/autodl-tmp/RNA-OmniDiffusion
    find . -maxdepth 1 ! -name dataset ! -name .env ! -name . -exec rm -rf {} + 2>/dev/null || true
    echo 'Remote cleaned'
"

# Restore dataset if needed
ssh "$REMOTE" "
    if [ ! -d /root/autodl-tmp/RNA-OmniDiffusion/dataset ] && [ -d /root/autodl-tmp/_backup/dataset ]; then
        cp -r /root/autodl-tmp/_backup/dataset /root/autodl-tmp/RNA-OmniDiffusion/dataset
        echo 'Dataset restored from backup'
    fi
    if [ ! -f /root/autodl-tmp/RNA-OmniDiffusion/.env ] && [ -f /root/autodl-tmp/_backup/.env ]; then
        cp /root/autodl-tmp/_backup/.env /root/autodl-tmp/RNA-OmniDiffusion/.env
        echo '.env restored from backup'
    fi
"

# Sync code
rsync -avz --delete \
    --exclude='.env' \
    --exclude='outputs/' \
    --exclude='dataset/' \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.sisyphus/' \
    README.md INDEX.md main.py \
    models/ scripts/ utils/ config/ docs/ \
    "$REMOTE:$REMOTE_PATH/"

echo "=== Sync complete ==="
echo ""
echo "Next steps on remote:"
echo "  ssh $REMOTE"
echo "  cd $REMOTE_PATH"
echo "  conda activate DL"
echo "  bash scripts/remote/run_all.sh check   # verify environment"
echo "  bash scripts/remote/run_all.sh all     # run all experiments"
