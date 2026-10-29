"""Text preprocessing backend: Rust when available, Python otherwise.

The server must never fail to start just because the Rust extension
was not built. Loading therefore tries ml_rust first and falls back to
the pure Python tokenizer on any failure, and the chosen backend is
reported so callers can tell which implementation served a request.

Both implementations expose the same shape: build_vocab, then
prepare_batch returning ids, an attention mask, and lengths. The Rust
side does the whole batch in one call; the Python side goes through
Preprocessor, which encodes one text at a time.
"""

import importlib.util

from app.preprocessing import PreparedBatch, Preprocessor
from app.tokenizer import Tokenizer

RUST_MODULE = "ml_rust"


def rust_available():
    return importlib.util.find_spec(RUST_MODULE) is not None


def _python_backend():
    return TextBackend(Tokenizer(), "python")


def _rust_backend():
    import ml_rust

    return TextBackend(ml_rust.Tokenizer(), "rust")


class TextBackend:
    def __init__(self, tokenizer, name):
        self._tokenizer = tokenizer
        self.name = name
        self._preprocessor = None if name == "rust" else Preprocessor(tokenizer)

    def build_vocab(self, texts, max_vocab=None, min_freq=1):
        return self._tokenizer.build_vocab(
            texts, max_vocab=max_vocab, min_freq=min_freq
        )

    @property
    def vocab_size(self):
        size = self._tokenizer.vocab_size
        return size() if callable(size) else size

    def prepare_batch(self, texts, max_length=None):
        if self.name == "rust":
            batch = self._tokenizer.prepare_batch(texts, max_length=max_length)
            return PreparedBatch(
                ids=[list(row) for row in batch.ids],
                attention_mask=[list(row) for row in batch.attention_mask],
                lengths=list(batch.lengths),
            )
        return self._preprocessor.prepare_batch(texts, max_length=max_length)


def load_backend(prefer="auto"):
    """Load a text backend.

    prefer="python" always returns the Python backend. prefer="rust"
    returns the Rust backend or raises ImportError when it is missing.
    prefer="auto" (the default) tries Rust and falls back to Python on
    any load failure, so serving never depends on the extension.
    """
    if prefer == "python":
        return _python_backend()
    if prefer == "rust":
        if not rust_available():
            raise ImportError("rust extension ml_rust is not built")
        return _rust_backend()
    try:
        if not rust_available():
            raise ImportError("rust extension ml_rust is not built")
        return _rust_backend()
    except Exception:
        return _python_backend()
