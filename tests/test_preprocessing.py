import importlib.util

import pytest

from app.benchmark import benchmark, per_item
from app.preprocessing import Preprocessor
from app.tokenizer import PAD_ID, Tokenizer

RUST_AVAILABLE = importlib.util.find_spec("ml_rust") is not None


def trained():
    tok = Tokenizer()
    tok.build_vocab(["hello world", "hello there"])
    return tok


def test_prepare_batch_pads_to_longest_and_builds_mask():
    prep = Preprocessor(trained())
    batch = prep.prepare_batch(["hello", "hello world"])
    assert batch.ids == [[2, PAD_ID], [2, 3]]
    assert batch.attention_mask == [[1, 0], [1, 1]]
    assert batch.lengths == [1, 2]


def test_prepare_batch_truncates_before_padding():
    prep = Preprocessor(trained())
    batch = prep.prepare_batch(["hello world there"], max_length=2)
    assert batch.ids == [[2, 3]]
    assert batch.lengths == [2]


def test_prepare_batch_handles_empty_batch_and_empty_text():
    prep = Preprocessor(trained())
    empty = prep.prepare_batch([])
    assert empty.ids == [] and empty.attention_mask == [] and empty.lengths == []
    batch = prep.prepare_batch(["", "hello"])
    assert batch.ids == [[PAD_ID], [2]]
    assert batch.attention_mask == [[0], [1]]
    assert batch.lengths == [0, 1]


def test_prepare_batch_uses_unk_for_missing_words():
    prep = Preprocessor(trained())
    batch = prep.prepare_batch(["hello nobody"])
    assert batch.ids == [[2, 1]]
    assert batch.attention_mask == [[1, 1]]


def test_benchmark_reports_runs_mean_and_best():
    result = benchmark(lambda: None, repeat=3)
    assert len(result["runs"]) == 3
    assert result["best_seconds"] <= result["mean_seconds"]
    assert all(run >= 0 for run in result["runs"])
    scaled = per_item(result, 2)
    assert scaled["best_us"] <= scaled["mean_us"]


def test_benchmark_rejects_bad_repeat_and_count():
    with pytest.raises(ValueError):
        benchmark(lambda: None, repeat=0)
    with pytest.raises(ValueError):
        per_item({"mean_seconds": 0, "best_seconds": 0}, 0)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="rust extension not built")
def test_rust_prepare_batch_matches_python():
    import ml_rust

    texts = ["hello", "hello world", "hello nobody"]
    py_batch = Preprocessor(trained()).prepare_batch(texts)
    rust_tok = ml_rust.Tokenizer()
    rust_tok.build_vocab(["hello world", "hello there"])
    rust_batch = rust_tok.prepare_batch(texts)
    assert rust_batch.ids == py_batch.ids
    assert rust_batch.attention_mask == py_batch.attention_mask
    assert rust_batch.lengths == py_batch.lengths


@pytest.mark.skipif(not RUST_AVAILABLE, reason="rust extension not built")
def test_rust_tokenizer_works_through_python_preprocessor():
    import ml_rust

    rust_tok = ml_rust.Tokenizer()
    rust_tok.build_vocab(["hello world", "hello there"])
    batch = Preprocessor(rust_tok).prepare_batch(["hello", "hello world"])
    assert batch.ids == [[2, PAD_ID], [2, 3]]
    assert batch.lengths == [1, 2]
