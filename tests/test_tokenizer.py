import importlib.util

import pytest

from app.tokenizer import PAD_ID, UNK_ID, Tokenizer

RUST_AVAILABLE = importlib.util.find_spec("ml_rust") is not None


def trained(lowercase=True):
    tok = Tokenizer(lowercase=lowercase)
    tok.build_vocab(["hello world", "hello there"])
    return tok


def test_tokenize_splits_words_and_punctuation():
    tok = Tokenizer()
    assert tok.tokenize("Hello, World! How are you?") == [
        "hello", ",", "world", "!", "how", "are", "you", "?",
    ]


def test_tokenize_keeps_case_when_disabled():
    tok = Tokenizer(lowercase=False)
    assert tok.tokenize("Hello World") == ["Hello", "World"]


def test_tokenize_ignores_extra_whitespace():
    tok = Tokenizer()
    assert tok.tokenize("  hello   world\n") == ["hello", "world"]


def test_vocab_orders_by_frequency_then_first_seen():
    tok = trained()
    assert tok.vocab["hello"] == 2
    assert tok.vocab["world"] == 3
    assert tok.vocab["there"] == 4
    assert tok.vocab_size == 5
    assert tok.vocab["<pad>"] == PAD_ID
    assert tok.vocab["<unk>"] == UNK_ID


def test_vocab_respects_min_freq_and_max_vocab():
    tok = Tokenizer()
    tok.build_vocab(["a b c", "a b"], min_freq=2)
    assert tok.vocab == {"<pad>": 0, "<unk>": 1, "a": 2, "b": 3}
    tok.build_vocab(["a b c", "a b"], max_vocab=3)
    assert tok.vocab_size == 3


def test_encode_uses_unk_for_missing_words():
    tok = trained()
    assert tok.encode("hello nobody") == [2, UNK_ID]


def test_encode_truncates_and_pads():
    tok = trained()
    assert tok.encode("hello world there", max_length=2) == [2, 3]
    assert tok.encode("hello", pad_to=3) == [2, PAD_ID, PAD_ID]
    assert tok.encode("hello world there", pad_to=2) == [2, 3]


def test_decode_round_trips_and_skips_padding():
    tok = Tokenizer()
    tok.build_vocab(["hello, world", "hello there"])
    ids = tok.encode("hello, world", pad_to=5)
    assert tok.decode(ids) == "hello, world"
    assert tok.decode(ids, skip_special=False) == "hello, world <pad> <pad>"


def test_decode_ignores_unknown_ids():
    tok = trained()
    assert tok.decode([2, 999]) == "hello"


@pytest.mark.skipif(not RUST_AVAILABLE, reason="rust extension not built")
def test_rust_tokenizer_matches_python():
    import ml_rust

    py_tok = trained()
    rust_tok = ml_rust.Tokenizer()
    rust_tok.build_vocab(["hello world", "hello there"])
    assert rust_tok.tokenize("Hello, World!") == py_tok.tokenize("Hello, World!")
    assert rust_tok.encode("hello nobody", pad_to=4) == py_tok.encode(
        "hello nobody", pad_to=4
    )
    assert rust_tok.decode([2, 3, 0]) == py_tok.decode([2, 3, 0])
    assert rust_tok.vocab_size == py_tok.vocab_size
