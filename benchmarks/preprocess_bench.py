"""Compare Python and Rust text preprocessing on a small corpus.

Builds a vocabulary from a few hundred short texts, then times three
ways of preparing a batch: the pure Python tokenizer, the Rust
tokenizer called once per text, and the Rust native batch call, which
does the whole batch in one trip across the Python/Rust boundary.

Run from the repository root:

    python benchmarks/preprocess_bench.py

If the ml_rust extension is not built, only the Python row is reported.
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.benchmark import benchmark, per_item  # noqa: E402
from app.preprocessing import Preprocessor  # noqa: E402
from app.tokenizer import Tokenizer  # noqa: E402

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models need clean, well prepared input data.",
    "Hello, world! How are you doing today?",
    "Batching groups requests together to cut down on overhead.",
    "Rust is fast, Python is flexible, and this server uses both.",
    "Attention masks tell the model which tokens are real input.",
]

NUM_TEXTS = 500
REPEAT = 5

RUST_AVAILABLE = importlib.util.find_spec("ml_rust") is not None


def build_corpus():
    texts = []
    while len(texts) < NUM_TEXTS:
        texts.extend(SENTENCES)
    return texts[:NUM_TEXTS]


def report(name, result, count):
    scaled = per_item(result, count)
    print(
        f"{name:<28} mean {result['mean_seconds'] * 1000:8.2f} ms/batch "
        f"({scaled['mean_us']:7.2f} us/text)  "
        f"best {result['best_seconds'] * 1000:8.2f} ms/batch "
        f"({scaled['best_us']:7.2f} us/text)"
    )
    return result


def main():
    texts = build_corpus()

    py_tok = Tokenizer()
    py_tok.build_vocab(texts)
    py_prep = Preprocessor(py_tok)

    print(f"preparing {len(texts)} texts, vocab size {py_tok.vocab_size}")
    print(f"{'implementation':<28} {'mean':>18}  {'best':>18}")
    py_result = report(
        "python",
        benchmark(lambda: py_prep.prepare_batch(texts), repeat=REPEAT),
        len(texts),
    )

    if not RUST_AVAILABLE:
        print("rust rows skipped: ml_rust extension is not built")
        print("(build it with maturin from the rust/ directory)")
        return

    import ml_rust

    rust_tok = ml_rust.Tokenizer()
    rust_tok.build_vocab(texts)
    sample = SENTENCES[0]
    if rust_tok.encode(sample) != py_tok.encode(sample):
        print("rust vocab does not match python vocab; rust rows skipped")
        return

    rust_prep = Preprocessor(rust_tok)
    report(
        "rust, one call per text",
        benchmark(lambda: rust_prep.prepare_batch(texts), repeat=REPEAT),
        len(texts),
    )
    rust_batch = report(
        "rust, one call per batch",
        benchmark(lambda: rust_tok.prepare_batch(texts), repeat=REPEAT),
        len(texts),
    )
    speedup = py_result["best_seconds"] / rust_batch["best_seconds"]
    print(f"rust batch prep is {speedup:.1f}x the python time (best runs)")


if __name__ == "__main__":
    main()
