from __future__ import annotations

import argparse
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer.pretokenization_example import find_chunk_boundaries
from cs336_basics.tokenizer.tokenizer import Tokenizer, train_bpe

FIXTURES_PATH = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
SPECIAL_TOKENS = ["<|endoftext|>"]


def train(args: argparse.Namespace) -> None:
    #profile_path = Path(".bpe_profile_input.txt")

    # text = args.src_path.read_text(encoding="utf-8")
    #profile_path.write_text(text[: args.chars], encoding="utf-8")

    # print(
    #     f"Profiling BPE on {args.chars:,} chars, "
    #     f"vocab_size={args.vocab_size}"
    # )
    start = time.perf_counter()

    vocab, merges = train_bpe(
        input_path=args.src_path,
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        output_dir=args.output_dir,
        num_processes=args.num_processes,
    )

    elapsed = time.perf_counter() - start

    print(f"Runtime: {elapsed:.3f}s")
    print(f"Vocab size: {len(vocab)}")
    print(f"Merges: {len(merges)}")
    print(f"Saved to: {args.output_dir}")


# Each encode worker loads the tokenizer once, then encodes many chunks with it.
_worker_tokenizer: Tokenizer | None = None


def _init_encode_worker(tokenizer_dir: str) -> None:
    global _worker_tokenizer
    _worker_tokenizer = Tokenizer.from_files(
        os.path.join(tokenizer_dir, "vocab.json"),
        os.path.join(tokenizer_dir, "merges.json"),
        SPECIAL_TOKENS,
    )


def _encode_chunk(task: tuple[str, int, int]) -> np.ndarray:
    path, start, end = task
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="replace")
    return np.fromiter(_worker_tokenizer.encode_iterable([text]), dtype=np.uint16)


def encode(args: argparse.Namespace) -> None:
    tokenizer = Tokenizer.from_files(
        args.tokenizer_dir / "vocab.json",
        args.tokenizer_dir / "merges.json",
        SPECIAL_TOKENS,
    )
    if len(tokenizer.vocab) > 2**16:
        raise ValueError(f"Vocab of {len(tokenizer.vocab)} tokens doesn't fit in uint16")

    path = str(args.src_path)
    num_chunks = max(args.num_processes, os.path.getsize(path) // (args.chunk_mb * 1024 * 1024))
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, SPECIAL_TOKENS[0].encode("utf-8"))
    tasks = [(path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    # The token count isn't known until the end, so stream raw uint16s to a temp file first.
    raw_path = args.output_path.with_suffix(".bin.tmp")
    start_time = time.perf_counter()
    num_tokens = 0
    with open(raw_path, "wb") as raw, multiprocessing.Pool(
        args.num_processes, initializer=_init_encode_worker, initargs=(str(args.tokenizer_dir),)
    ) as pool:
        # imap (not imap_unordered) so chunks are written in file order.
        for i, ids in enumerate(pool.imap(_encode_chunk, tasks), start=1):
            ids.tofile(raw)
            num_tokens += len(ids)
            print(f"\r{i}/{len(tasks)} chunks, {num_tokens:,} tokens", end="", flush=True)
    print()

    # Copy into a real .npy so it can be loaded with np.load(path, mmap_mode="r").
    raw_ids = np.memmap(raw_path, dtype=np.uint16, mode="r", shape=(num_tokens,))
    out = np.lib.format.open_memmap(args.output_path, mode="w+", dtype=np.uint16, shape=(num_tokens,))
    copy_block = 100_000_000
    for i in range(0, num_tokens, copy_block):
        out[i : i + copy_block] = raw_ids[i : i + copy_block]
    out.flush()
    del out, raw_ids
    raw_path.unlink()

    elapsed = time.perf_counter() - start_time
    src_bytes = os.path.getsize(path)
    print(f"Runtime: {elapsed:.3f}s")
    print(f"Tokens: {num_tokens:,} ({src_bytes / max(num_tokens, 1):.2f} bytes/token)")
    print(f"Saved to: {args.output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a BPE tokenizer and save vocab.json/merges.json.")
    train_parser.add_argument(
        "--chars",
        type=int,
        default=None,
        help="Number of characters to profile.",
    )
    train_parser.add_argument(
        "--vocab-size",
        type=int,
        default=700,
    )
    train_parser.add_argument(
        "--src-path",
        type=Path,
        default=FIXTURES_PATH / "tinystories_sample_5M.txt",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to write vocab.json and merges.json to.",
    )
    train_parser.add_argument("--num-processes", type=int, default=12)
    train_parser.set_defaults(func=train)

    encode_parser = subparsers.add_parser("encode", help="Encode a text file into a uint16 .npy of token IDs.")
    encode_parser.add_argument("--tokenizer-dir", type=Path, required=True, help="Directory with vocab.json and merges.json.")
    encode_parser.add_argument("--src-path", type=Path, required=True)
    encode_parser.add_argument("--output-path", type=Path, required=True, help="Where to write the .npy file.")
    encode_parser.add_argument("--num-processes", type=int, default=12)
    encode_parser.add_argument("--chunk-mb", type=int, default=16, help="Size of the text chunk each worker encodes at a time.")
    encode_parser.set_defaults(func=encode)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
