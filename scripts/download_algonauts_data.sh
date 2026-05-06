#!/usr/bin/env bash
#
# Download the Algonauts 2025 challenge dataset on EC2.
#
# DO NOT RUN ON LOCAL MAC. The dataset is ~100 GB. Run on the
# blip2-validation instance (i-0bdbdf83c4db9bdae) where there's room
# on the EBS volume.
#
# Usage (from local laptop):
#   aws ec2 start-instances --instance-ids i-0bdbdf83c4db9bdae --region us-east-2
#   aws ec2 wait instance-running --instance-ids i-0bdbdf83c4db9bdae --region us-east-2
#   IP=$(aws ec2 describe-instances --instance-ids i-0bdbdf83c4db9bdae \
#        --region us-east-2 \
#        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
#   scp -i quest-kartik-personal.pem unified/scripts/download_algonauts_data.sh \
#       ubuntu@$IP:~/
#   ssh -i quest-kartik-personal.pem ubuntu@$IP \
#       'bash ~/download_algonauts_data.sh 2>&1 | tee ~/algonauts_download.log'
#
# Estimated time: 1–6 hours depending on subject coverage and bandwidth.
# Estimated EBS usage: ~100 GB (full dataset) or ~30 GB (one subject).

set -euo pipefail

DATA_ROOT="${ALGONAUTS_DATA_ROOT:-$HOME/algonauts_2025}"
SUBJECTS="${ALGONAUTS_SUBJECTS:-01 02 03 05}"
SKIP_STIMULI="${ALGONAUTS_SKIP_STIMULI:-0}"

echo "=== Algonauts 2025 download ==="
echo "Data root:        $DATA_ROOT"
echo "Subjects to fetch: $SUBJECTS"
echo "Skip stimuli:     $SKIP_STIMULI"
echo

# 1. Install DataLad if missing (uses pipx for an isolated install)
if ! command -v datalad >/dev/null 2>&1; then
    echo "Installing DataLad..."
    if ! command -v pipx >/dev/null 2>&1; then
        sudo apt-get update -qq && sudo apt-get install -y -qq pipx git-annex
        pipx ensurepath
        export PATH="$HOME/.local/bin:$PATH"
    fi
    pipx install datalad
    pipx inject datalad datalad-osf  # for OSF-hosted subdatasets
fi

datalad --version | head -1

# 2. Install the dataset metadata (no downloads yet)
mkdir -p "$DATA_ROOT"
cd "$(dirname "$DATA_ROOT")"
if [ ! -d "$DATA_ROOT/.datalad" ]; then
    echo "Installing DataLad dataset (metadata only, no large files)..."
    datalad install -r -s \
        https://github.com/courtois-neuromod/algonauts_2025.competitors.git \
        "$(basename "$DATA_ROOT")"
fi
cd "$DATA_ROOT"

# 3. Download stimuli (videos + transcripts). ~30 GB.
if [ "$SKIP_STIMULI" != "1" ]; then
    echo
    echo "Fetching stimuli (videos + transcripts)..."
    datalad get -r -J 8 stimuli/ || {
        echo "Some stimuli failed to fetch — partial dataset still usable."
    }
fi

# 4. Download fMRI per subject. ~15 GB per subject for full Friends + Movie10.
for sub in $SUBJECTS; do
    echo
    echo "Fetching fMRI for sub-$sub..."
    if [ -d "fmri/sub-$sub" ]; then
        datalad get -r -J 8 "fmri/sub-$sub/" || {
            echo "Some files for sub-$sub failed; continuing."
        }
    else
        echo "Warning: fmri/sub-$sub not present; skipping."
    fi
done

# 5. Summary
echo
echo "=== Done. Final usage: ==="
du -sh "$DATA_ROOT" 2>/dev/null
echo
echo "Stimuli:"
ls "$DATA_ROOT/stimuli" 2>/dev/null
echo "fMRI subjects present:"
ls "$DATA_ROOT/fmri" 2>/dev/null
