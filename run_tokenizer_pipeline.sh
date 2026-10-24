#!/usr/bin/env bash
# Train the TinyStories and OpenWebText tokenizers, then encode each dataset's
# train and valid splits into uint16 .npy files with its own tokenizer.
#
# Usage: ./run_tokenizer_pipeline.sh [num_processes]   (default 12)
# Output is also logged to data/pipeline.log.
set -euo pipefail

cd "$(dirname "$0")"

NUM_PROCESSES="${1:-12}"
DATA=data
TOKENIZERS="$DATA/tokenizers"
ENCODED="$DATA/encoded"
TOOL="uv run python -m cs336_basics.tokenizer.profile_tokenizer"

exec > >(tee -a "$DATA/pipeline.log") 2>&1

step() {
    echo
    echo "=== [$(date '+%H:%M:%S')] $1 ==="
}

train() {
    local src="$1" vocab_size="$2" out_dir="$3"
    $TOOL train --src-path "$src" --vocab-size "$vocab_size" \
        --output-dir "$out_dir" --num-processes "$NUM_PROCESSES"
}

encode() {
    local tokenizer_dir="$1" src="$2" out="$3"
    $TOOL encode --tokenizer-dir "$tokenizer_dir" --src-path "$src" \
        --output-path "$out" --num-processes "$NUM_PROCESSES"
}

start=$SECONDS

step "2. Train TinyStories tokenizer (10k vocab)"
train "$DATA/TinyStoriesV2-GPT4-train.txt" 10000 "$TOKENIZERS/tinystories_10k"

step "3a. Decompress OpenWebText"
if [[ -f "$DATA/owt_train.txt" ]]; then
    echo "$DATA/owt_train.txt already exists, skipping"
else
    # Decompress to a temp name so an interrupted run doesn't leave a truncated file behind.
    gunzip -c "$DATA/owt_train.txt.gz" > "$DATA/owt_train.txt.partial"
    mv "$DATA/owt_train.txt.partial" "$DATA/owt_train.txt"
fi

step "3b. Train OpenWebText tokenizer (32k vocab)"
train "$DATA/owt_train.txt" 32000 "$TOKENIZERS/owt_32k"

step "4. Encode TinyStories"
encode "$TOKENIZERS/tinystories_10k" "$DATA/TinyStoriesV2-GPT4-train.txt" "$ENCODED/tinystories_train.npy"
encode "$TOKENIZERS/tinystories_10k" "$DATA/TinyStoriesV2-GPT4-valid.txt" "$ENCODED/tinystories_valid.npy"

step "4. Encode OpenWebText"
encode "$TOKENIZERS/owt_32k" "$DATA/owt_train.txt" "$ENCODED/owt_train.npy"
encode "$TOKENIZERS/owt_32k" "$DATA/owt_valid.txt" "$ENCODED/owt_valid.npy"

elapsed=$((SECONDS - start))
step "Done in $((elapsed / 60))m $((elapsed % 60))s"
