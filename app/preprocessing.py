"""Text preprocessing: turn raw texts into a padded batch for a model.

A model wants numbers, not strings, and it wants them in a rectangle:
every sequence in a batch padded to the same length. This module does
that step. It works with either tokenizer implementation, the pure
Python one in app/tokenizer.py or the Rust one in ml_rust, because both
expose the same encode method.

Padding uses id 0, the same reserved padding id the tokenizers use.
The attention mask marks real tokens with 1 and padding with 0, so a
model can tell the two apart instead of treating padding as input.
"""

from dataclasses import dataclass

from app.tokenizer import PAD_ID


@dataclass
class PreparedBatch:
    ids: list
    attention_mask: list
    lengths: list


class Preprocessor:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def prepare_batch(self, texts, max_length=None):
        encoded = [
            self._tokenizer.encode(text, max_length=max_length) for text in texts
        ]
        width = max((len(ids) for ids in encoded), default=0)
        ids = [row + [PAD_ID] * (width - len(row)) for row in encoded]
        attention_mask = [
            [1] * len(row) + [0] * (width - len(row)) for row in encoded
        ]
        return PreparedBatch(
            ids=ids,
            attention_mask=attention_mask,
            lengths=[len(row) for row in encoded],
        )
