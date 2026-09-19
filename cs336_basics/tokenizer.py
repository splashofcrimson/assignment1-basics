from typing import NamedTuple
from pathlib import Path
import regex as re

class BPE_CFG(NamedTuple):
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: list[bytes] = [bytes([i]) for i in range(256)] + [special_token.encode('utf-8') for special_token in special_tokens]
    return dict(enumerate(vocab))

def pretokenize(input: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    generic = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    special = "|".join(re.escape(special_token) for special_token in sorted(special_tokens, key=len, reverse=True))
    input_re = re.finditer(rf"(?P<special>{special})|(?P<generic>{generic})", input)
    freq_table: dict[tuple[bytes, ...], int] = {}
    for pretoken in input_re:
        if pretoken.group("special"):
            continue

        pretoken_bytes = tuple(bytes([b]) for b in pretoken.group().encode('utf-8'))
        freq_table[pretoken_bytes] = freq_table.get(pretoken_bytes, 0) + 1
    return freq_table

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> BPE_CFG:
    vocab: dict[int, bytes] = init_vocab(special_tokens)

    input: str = Path(input_path).read_text(encoding="utf-8")
    freq_table: dict[tuple[bytes, ...], int] = pretokenize(input, special_tokens)

    merges: list[tuple[bytes, bytes]] = []
    while(len(vocab.keys()) < vocab_size):

        # Find highest frequency pairs
        pair_freq_table: dict[tuple[bytes, bytes], int] = {}
        best_pair: tuple[tuple[bytes, bytes], int] = ()
        for _, (pretoken, pretoken_freq) in enumerate(freq_table.items()):
            for pretoken_byte_idx in range(len(pretoken) - 1):
                byte_pair: tuple[bytes, bytes] = (pretoken[pretoken_byte_idx], pretoken[pretoken_byte_idx + 1])
                pair_freq_table[byte_pair] = pair_freq_table.get(byte_pair, 0) + pretoken_freq
                if not best_pair or pair_freq_table[byte_pair] > best_pair[1] or (pair_freq_table[byte_pair] == best_pair[1] and byte_pair > best_pair[0]):
                    best_pair = (byte_pair, pair_freq_table[byte_pair])
        
        # Merge pre-tokens
        temp_freq_table: dict[tuple[bytes, ...], int] = {}
        pair: tuple[bytes, bytes] = best_pair[0]
        for _, (pretoken, _) in enumerate(freq_table.items()):
            merged_pretoken: list[bytes] = []
            pretoken_byte_idx: int = 0
            while pretoken_byte_idx < len(pretoken):
                pretoken_byte = pretoken[pretoken_byte_idx]
                if pretoken_byte == pair[0] and pretoken_byte_idx < (len(pretoken) - 1) and pretoken[pretoken_byte_idx + 1] == pair[1]:
                    merged_pretoken.append(pair[0] + pair[1])
                    pretoken_byte_idx += 2
                else:
                    merged_pretoken.append(pretoken_byte)
                    pretoken_byte_idx += 1
            temp_freq_table[tuple(merged_pretoken)] = temp_freq_table.get(tuple(merged_pretoken), 0) + freq_table[pretoken]

        vocab[len(vocab.keys())] = pair[0] + pair[1]
        merges.append(pair)
        freq_table = temp_freq_table

    return vocab, merges
