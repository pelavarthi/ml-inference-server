"""Word tokenizer with a small learned vocabulary.

This is the reference implementation of the tokenizer that the Rust
extension in rust/ mirrors. The server uses one of the two
implementations through app/backend.py, which falls back to this one
when the Rust extension is missing.

Tokenizing rules: words and numbers stay whole, punctuation becomes its
own token, and runs of whitespace just separate tokens. Two ids are
reserved: 0 for padding and 1 for words outside the vocabulary.
"""

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1


def _is_punctuation(char):
    return not char.isalnum() and not char.isspace()


class Tokenizer:
    def __init__(self, lowercase=True):
        self.lowercase = lowercase
        self._vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        self._tokens = {PAD_ID: PAD_TOKEN, UNK_ID: UNK_TOKEN}

    @property
    def vocab(self):
        return dict(self._vocab)

    @property
    def vocab_size(self):
        return len(self._vocab)

    def tokenize(self, text):
        if self.lowercase:
            text = text.lower()
        tokens = []
        current = []
        for char in text:
            if char.isalnum():
                current.append(char)
            else:
                if current:
                    tokens.append("".join(current))
                    current = []
                if _is_punctuation(char):
                    tokens.append(char)
        if current:
            tokens.append("".join(current))
        return tokens

    def build_vocab(self, texts, max_vocab=None, min_freq=1):
        counts = {}
        first_seen = {}
        for text in texts:
            for token in self.tokenize(text):
                if token not in counts:
                    counts[token] = 0
                    first_seen[token] = len(first_seen)
                counts[token] += 1
        candidates = [t for t in counts if counts[t] >= min_freq]
        candidates.sort(key=lambda t: (-counts[t], first_seen[t]))
        if max_vocab is not None:
            candidates = candidates[: max(0, max_vocab - 2)]
        self._vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        self._tokens = {PAD_ID: PAD_TOKEN, UNK_ID: UNK_TOKEN}
        for index, token in enumerate(candidates, start=2):
            self._vocab[token] = index
            self._tokens[index] = token
        return self.vocab

    def encode(self, text, max_length=None, pad_to=None):
        ids = [self._vocab.get(token, UNK_ID) for token in self.tokenize(text)]
        if max_length is not None:
            ids = ids[:max_length]
        if pad_to is not None:
            ids = ids[:pad_to]
            ids = ids + [PAD_ID] * (pad_to - len(ids))
        return ids

    def decode(self, ids, skip_special=True):
        tokens = []
        for token_id in ids:
            token = self._tokens.get(token_id)
            if token is None:
                continue
            if skip_special and token in (PAD_TOKEN, UNK_TOKEN):
                continue
            tokens.append(token)
        parts = []
        for token in tokens:
            if len(token) == 1 and _is_punctuation(token):
                parts.append(token)
            else:
                if parts:
                    parts.append(" ")
                parts.append(token)
        return "".join(parts)
