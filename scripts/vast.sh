#!/usr/bin/env bash
# vast.sh — convenience wrapper around the vastai CLI for this project.
#
# Usage:
#   bash scripts/vast.sh search                  # show some good offers
#   bash scripts/vast.sh up <OFFER_ID>           # create instance + save ID to .vast_id
#   bash scripts/vast.sh ssh                     # ssh into the saved instance
#   bash scripts/vast.sh push                    # rsync code Mac → instance
#   bash scripts/vast.sh pull                    # rsync results instance → Mac
#   bash scripts/vast.sh stop                    # halt GPU billing, keep disk
#   bash scripts/vast.sh start                   # resume
#   bash scripts/vast.sh destroy                 # nuke (don't do this casually)
#   bash scripts/vast.sh status                  # show instance status
#
# Requires:
#   - vastai CLI authenticated (vastai set api-key …)
#   - SSH key registered on vast.ai

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ID_FILE="$REPO_ROOT/.vast_id"

# Tweakable defaults
DISK_GB="${VAST_DISK_GB:-80}"
IMAGE="${VAST_IMAGE:-pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime}"
REMOTE_PATH="${VAST_REMOTE_PATH:-/workspace/cacheblend-hf}"

ONSTART_CMD='cat <<EOF >> /etc/environment
HF_HOME=/workspace/hf_cache
HF_HUB_CACHE=/workspace/hf_cache/hub
TRANSFORMERS_CACHE=/workspace/hf_cache/transformers
EOF
mkdir -p /workspace/hf_cache
echo "ready"'

cmd="${1:-help}"

require_id() {
    if [[ ! -f "$ID_FILE" ]]; then
        echo "ERROR: no .vast_id file. Run 'vast.sh up <OFFER_ID>' first." >&2
        exit 1
    fi
    INSTANCE_ID="$(cat "$ID_FILE")"
}

case "$cmd" in
    search)
        # A reasonable default search. Tweak in VAST_GUIDE.md
        vastai search offers \
            'reliability>0.99 num_gpus=1 gpu_ram>=24 cuda_max_good>=12.1 inet_down>=100 verified=True' \
            -o 'dph_total' | head -30
        ;;

    up)
        offer_id="${2:-}"
        if [[ -z "$offer_id" ]]; then
            echo "Usage: vast.sh up <OFFER_ID>" >&2
            exit 1
        fi
        echo "Creating instance from offer $offer_id (disk ${DISK_GB}GB, image $IMAGE)…"
        out=$(vastai create instance "$offer_id" \
            --image "$IMAGE" \
            --disk "$DISK_GB" \
            --ssh --direct \
            --onstart-cmd "$ONSTART_CMD")
        echo "$out"
        # Extract new_contract from output (it's JSON-like)
        new_id=$(echo "$out" | grep -oE 'new_contract[^0-9]*[0-9]+' | grep -oE '[0-9]+' | head -1)
        if [[ -z "$new_id" ]]; then
            echo "WARNING: could not auto-detect instance ID. Run 'vastai show instances' and write the ID to $ID_FILE manually." >&2
        else
            echo "$new_id" > "$ID_FILE"
            echo "Instance ID $new_id saved to .vast_id"
            echo "Wait ~30-60s for boot, then: bash scripts/vast.sh ssh"
        fi
        ;;

    status)
        require_id
        vastai show instance "$INSTANCE_ID"
        ;;

    ssh)
        require_id
        url=$(vastai ssh-url "$INSTANCE_ID")
        echo "Connecting: $url"
        eval "$url"
        ;;

    push)
        require_id
        url=$(vastai ssh-url "$INSTANCE_ID")
        # Parse "ssh -p PORT user@HOST"
        port=$(echo "$url" | grep -oE 'ssh -p [0-9]+' | awk '{print $3}')
        target=$(echo "$url" | grep -oE '[a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+' | tail -1)
        echo "Syncing $REPO_ROOT/  →  $target:$REMOTE_PATH/"
        rsync -avz --delete \
            --exclude '.venv' --exclude '.git' --exclude '__pycache__' \
            --exclude 'external' --exclude '.vast_id' --exclude '.env' \
            --exclude 'reports/phase-*-attachments' \
            --exclude '*.pt' --exclude '*.bin' --exclude '*.safetensors' \
            -e "ssh -p $port" \
            "$REPO_ROOT/" "$target:$REMOTE_PATH/"
        ;;

    pull)
        require_id
        url=$(vastai ssh-url "$INSTANCE_ID")
        port=$(echo "$url" | grep -oE 'ssh -p [0-9]+' | awk '{print $3}')
        target=$(echo "$url" | grep -oE '[a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+' | tail -1)
        echo "Pulling reports/ and benchmarks/results/ from $target"
        rsync -avz -e "ssh -p $port" \
            "$target:$REMOTE_PATH/reports/" "$REPO_ROOT/reports/"
        rsync -avz -e "ssh -p $port" \
            "$target:$REMOTE_PATH/benchmarks/results/" "$REPO_ROOT/benchmarks/results/" 2>/dev/null || true
        ;;

    stop)
        require_id
        vastai stop instance "$INSTANCE_ID"
        echo "Stopped. Disk preserved. GPU billing halted."
        ;;

    start)
        require_id
        vastai start instance "$INSTANCE_ID"
        echo "Starting. Wait ~30s, then: bash scripts/vast.sh ssh"
        ;;

    destroy)
        require_id
        echo "About to DESTROY instance $INSTANCE_ID. This deletes everything including /workspace."
        read -p "Type 'yes' to confirm: " confirm
        if [[ "$confirm" == "yes" ]]; then
            vastai destroy instance "$INSTANCE_ID"
            rm -f "$ID_FILE"
            echo "Destroyed."
        else
            echo "Aborted."
        fi
        ;;

    help|*)
        sed -n '2,17p' "$0"
        ;;
esac
