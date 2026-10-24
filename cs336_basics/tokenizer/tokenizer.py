import json
import multiprocessing
import os
from collections import Counter, defaultdict
from heapq import heapify, heappop, heappush
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple, Optional

import regex as re

from cs336_basics.tokenizer.pretokenization_example import find_chunk_boundaries


Pair = tuple[bytes, bytes]
Pretoken = tuple[bytes, ...]


class Tokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Reverse vocab lookup, and each merge's position in the merge list (lower = applied first).
        self.bytes_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self.merge_ranks = {merge: rank for rank, merge in enumerate(merges)}
        # Pretokens repeat a lot in real text, so remember the IDs for each one we've encoded.
        # Cleared when it gets large so encoding a big corpus doesn't grow memory without bound.
        self.cache: dict[Pretoken, list[int]] = {}
        self.max_cache_size = 1_000_000

        # Map each special token to its ID, adding it to the vocab if it isn't already there.
        self.special_token_ids = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            token_id = self.bytes_to_id.get(token_bytes)
            if token_id is None:
                token_id = max(self.vocab, default=-1) + 1
                self.vocab[token_id] = token_bytes
                self.bytes_to_id[token_bytes] = token_id
            self.special_token_ids[token] = token_id

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        # Files are written by save_bpe: token IDs as string keys, tokens as hex strings.
        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            vocab_json = json.load(vocab_file)
        with open(merges_filepath, encoding="utf-8") as merges_file:
            merges_json = json.load(merges_file)

        vocab = {int(token_id): bytes.fromhex(token) for token_id, token in vocab_json.items()}
        merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in merges_json]
        return cls(vocab, merges, special_tokens)

    def pretokenize(self, text: str) -> Iterator[Pretoken | str]:
        """Yield the pretokens of text in order. Special tokens are kept whole, as their original str."""
        generic = (
                r"'(?:[sdmt]|ll|ve|re)"
                r"| ?\p{L}+"
                r"| ?\p{N}+"
                r"| ?[^\s\p{L}\p{N}]+"
                r"|\s+(?!\S)"
                r"|\s+"
            )
        
        if self.special_tokens:
            escaped_tokens = (
                re.escape(token)
                for token in sorted(self.special_tokens, key=len, reverse=True)
            )
            special_pattern = "(" + "|".join(escaped_tokens) + ")"
            chunks = re.split(special_pattern, text)
            special_set = set(self.special_tokens)
        else:
            chunks = [text]
            special_set = set()

        for chunk in chunks:
            if chunk in special_set:
                yield chunk
                continue
            for match in re.finditer(generic, chunk):
                yield tuple(bytes([b]) for b in match.group().encode("utf-8"))

    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable([text]))

    def encode_pretoken(self, pretoken: Pretoken) -> list[int]:
        # Applying merges in training order is the same as repeatedly applying the
        # lowest-ranked merge present, but this only looks at pairs actually in the pretoken.
        pretoken = list(pretoken)
        while len(pretoken) > 1:
            pairs = zip(pretoken, pretoken[1:])
            merge = min(pairs, key=lambda pair: self.merge_ranks.get(pair, float("inf")))
            if merge not in self.merge_ranks:
                break
            pretoken_idx = 0
            merged_pretoken = []
            while pretoken_idx < len(pretoken):
                if pretoken_idx < len(pretoken) - 1 and (pretoken[pretoken_idx], pretoken[pretoken_idx + 1]) == merge:
                    merged_pretoken.append(merge[0] + merge[1])
                    pretoken_idx += 2
                else:
                    merged_pretoken.append(pretoken[pretoken_idx])
                    pretoken_idx += 1
            pretoken = merged_pretoken
        return [self.bytes_to_id[token] for token in pretoken]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for iterable_chunk in iterable:
            for pretoken in self.pretokenize(iterable_chunk):
                if isinstance(pretoken, str):
                    yield self.special_token_ids[pretoken]
                    continue
                if pretoken not in self.cache:
                    if len(self.cache) >= self.max_cache_size:
                        self.cache.clear()
                    self.cache[pretoken] = self.encode_pretoken(pretoken)
                yield from self.cache[pretoken]

    def decode(self, ids: list[int]) -> str:
        # A multi-byte character can be split across tokens, so join all bytes before decoding.
        decoded_bytes = b"".join(self.vocab[id] for id in ids)
        return decoded_bytes.decode("utf-8", errors="replace")

class BPE_CFG(NamedTuple):
    vocab: dict[int, bytes]
    merges: list[Pair]


def save_bpe(cfg: BPE_CFG, output_dir: str | Path) -> None:
    """Save a BPE vocabulary and ordered merge rules as lossless JSON files.

    JSON does not support bytes, so each token is represented by a hex string.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_json = {str(token_id): token.hex() for token_id, token in cfg.vocab.items()}
    merges_json = [[left.hex(), right.hex()] for left, right in cfg.merges]

    (output_dir / "vocab.json").write_text(
        json.dumps(vocab_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "merges.json").write_text(
        json.dumps(merges_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_bpe(output_dir: str | Path) -> BPE_CFG:
    """Reconstruct the original bytes and merge order from saved JSON files."""
    output_dir = Path(output_dir)
    vocab_json = json.loads((output_dir / "vocab.json").read_text(encoding="utf-8"))
    merges_json = json.loads((output_dir / "merges.json").read_text(encoding="utf-8"))

    vocab = {int(token_id): bytes.fromhex(token) for token_id, token in vocab_json.items()}
    merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in merges_json]
    return BPE_CFG(vocab=vocab, merges=merges)


class HeapItem:
    """A max-heap entry ordered by weighted frequency, then BPE tie-breaker."""

    __slots__ = ("frequency", "pair")

    def __init__(self, frequency: int, pair: Pair):
        self.frequency = frequency
        self.pair = pair

    def __lt__(self, other: "HeapItem") -> bool:
        # heapq is a min-heap: reverse both comparisons to select the
        # highest frequency and lexicographically greatest pair.
        return (self.frequency, self.pair) > (other.frequency, other.pair)


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = [bytes([i]) for i in range(256)]
    vocab.extend(token.encode("utf-8") for token in special_tokens)
    return dict(enumerate(vocab))


def pretokenize(text: str, special_tokens: list[str]) -> dict[Pretoken, int]:
    generic = (
        r"'(?:[sdmt]|ll|ve|re)"
        r"| ?\p{L}+"
        r"| ?\p{N}+"
        r"| ?[^\s\p{L}\p{N}]+"
        r"|\s+(?!\S)"
        r"|\s+"
    )

    if special_tokens:
        escaped_tokens = (
            re.escape(token)
            for token in sorted(special_tokens, key=len, reverse=True)
        )
        special_pattern = "(" + "|".join(escaped_tokens) + ")"
        chunks = re.split(special_pattern, text)
        special_set = set(special_tokens)
    else:
        chunks = [text]
        special_set = set()

    counts: Counter[Pretoken] = Counter()
    for chunk in chunks:
        if chunk in special_set:
            continue
        for match in re.finditer(generic, chunk):
            pretoken = tuple(bytes([b]) for b in match.group().encode("utf-8"))
            counts[pretoken] += 1

    return dict(counts)


def process_chunk(args: tuple[str, int, int, list[str]]) -> dict[Pretoken, int]:
    path, start, end, special_tokens = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")
    return pretokenize(text, special_tokens)


def parallel_pretokenize(
    path: str | Path,
    special_tokens: list[str],
    num_processes: int = 12,
    chunk_bytes: int = 64 * 1024 * 1024,
) -> dict[Pretoken, int]:
    path = str(path)
    # Many small chunks keep each worker's memory bounded; the pool hands them out as workers free up.
    num_chunks = max(num_processes, os.path.getsize(path) // chunk_bytes)
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")

    tasks = [
        (path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    total: Counter[Pretoken] = Counter()
    with multiprocessing.Pool(processes=num_processes) as pool:
        for counts in pool.imap_unordered(process_chunk, tasks):
            total.update(counts)

    return dict(total)


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
    output_dir: str | Path | None = None,
    num_processes: int = 12,
) -> BPE_CFG:
    vocab = init_vocab(special_tokens)
    if vocab_size < len(vocab):
        raise ValueError("vocab_size is smaller than the initial vocabulary")

    # List indices are permanent pretoken IDs; frequencies never change.
    freq_table: list[tuple[Pretoken, int]] = list(
        parallel_pretokenize(input_path, special_tokens, num_processes).items()
    )

    pair_freq_table: Counter[Pair] = Counter()
    pair_to_pretoken_ids: dict[Pair, set[int]] = defaultdict(set)
    merges: list[Pair] = []

    for pretoken_id, (pretoken, frequency) in enumerate(freq_table):
        local_pairs = Counter(zip(pretoken, pretoken[1:]))
        for pair, occurrences in local_pairs.items():
            pair_freq_table[pair] += occurrences * frequency
            pair_to_pretoken_ids[pair].add(pretoken_id)

    # Build the heap once; subsequent updates append newer snapshots.
    # Old snapshots are discarded lazily when they reach the heap's top.
    heap = [HeapItem(count, pair) for pair, count in pair_freq_table.items()]
    heapify(heap)

    while len(vocab) < vocab_size and pair_freq_table:
        # Pop until we find an entry that matches the pair's current count.
        while heap:
            entry = heappop(heap)
            if pair_freq_table.get(entry.pair, 0) == entry.frequency:
                best_pair = entry.pair
                break
        else:
            raise RuntimeError("No valid BPE pair in heap")

        affected_ids = tuple(pair_to_pretoken_ids[best_pair])
        changed_pairs: set[Pair] = set()

        for pretoken_id in affected_ids:
            pretoken, frequency = freq_table[pretoken_id]

            merged: list[bytes] = []
            i = 0
            while i < len(pretoken):
                if i + 1 < len(pretoken) and (pretoken[i], pretoken[i + 1]) == best_pair:
                    merged.append(pretoken[i] + pretoken[i + 1])
                    i += 2
                else:
                    merged.append(pretoken[i])
                    i += 1
            new_pretoken = tuple(merged)

            # Update only pair counts within pretokens containing best_pair.
            old_pairs = Counter(zip(pretoken, pretoken[1:]))
            new_pairs = Counter(zip(new_pretoken, new_pretoken[1:]))

            for pair in old_pairs.keys() | new_pairs.keys():
                old_count = old_pairs[pair]
                new_count = new_pairs[pair]
                delta = (new_count - old_count) * frequency

                if delta:
                    pair_freq_table[pair] += delta
                    changed_pairs.add(pair)
                    if pair_freq_table[pair] == 0:
                        del pair_freq_table[pair]

                if old_count > 0 and new_count == 0:
                    ids = pair_to_pretoken_ids[pair]
                    ids.remove(pretoken_id)
                    if not ids:
                        del pair_to_pretoken_ids[pair]
                elif old_count == 0 and new_count > 0:
                    pair_to_pretoken_ids[pair].add(pretoken_id)

            # Replace the sequence in place; preserve ID and corpus frequency.
            freq_table[pretoken_id] = (new_pretoken, frequency)

        # Only push each modified pair once, with its final frequency.
        # Absent/zero-count pairs are excluded.
        for pair in changed_pairs:
            if pair in pair_freq_table:
                heappush(heap, HeapItem(pair_freq_table[pair], pair))

        # Rebuild occasionally to bound memory consumed by stale entries.
        if len(heap) > max(4096, 4 * len(pair_freq_table)):
            heap = [HeapItem(count, pair) for pair, count in pair_freq_table.items()]
            heapify(heap)

        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merges.append(best_pair)

    cfg = BPE_CFG(vocab=vocab, merges=merges)
    if output_dir is not None:
        save_bpe(cfg, output_dir)
    return cfg
