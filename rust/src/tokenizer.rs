//! Word tokenizer: the same rules as `app/tokenizer.py`.
//!
//! Words and numbers stay whole, punctuation becomes its own token, and
//! whitespace only separates tokens. Id 0 is padding, id 1 is unknown.

use std::collections::HashMap;

pub const PAD_TOKEN: &str = "<pad>";
pub const UNK_TOKEN: &str = "<unk>";
pub const PAD_ID: usize = 0;
pub const UNK_ID: usize = 1;

pub struct Tokenizer {
    pub lowercase: bool,
    vocab: HashMap<String, usize>,
    tokens: HashMap<usize, String>,
}

fn is_punctuation(c: char) -> bool {
    !c.is_alphanumeric() && !c.is_whitespace()
}

impl Tokenizer {
    pub fn new(lowercase: bool) -> Self {
        let mut vocab = HashMap::new();
        let mut tokens = HashMap::new();
        vocab.insert(PAD_TOKEN.to_string(), PAD_ID);
        vocab.insert(UNK_TOKEN.to_string(), UNK_ID);
        tokens.insert(PAD_ID, PAD_TOKEN.to_string());
        tokens.insert(UNK_ID, UNK_TOKEN.to_string());
        Tokenizer { lowercase, vocab, tokens }
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let lowered;
        let text = if self.lowercase {
            lowered = text.to_lowercase();
            lowered.as_str()
        } else {
            text
        };
        let mut tokens = Vec::new();
        let mut current = String::new();
        for c in text.chars() {
            if c.is_alphanumeric() {
                current.push(c);
            } else {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                if is_punctuation(c) {
                    tokens.push(c.to_string());
                }
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        tokens
    }

    pub fn build_vocab(
        &mut self,
        texts: &[String],
        max_vocab: Option<usize>,
        min_freq: usize,
    ) -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut first_seen: HashMap<String, usize> = HashMap::new();
        for text in texts {
            for token in self.tokenize(text) {
                if !counts.contains_key(&token) {
                    first_seen.insert(token.clone(), first_seen.len());
                }
                *counts.entry(token).or_insert(0) += 1;
            }
        }
        let mut candidates: Vec<String> = counts
            .iter()
            .filter(|(_, &count)| count >= min_freq)
            .map(|(token, _)| token.clone())
            .collect();
        candidates.sort_by_key(|token| {
            (
                usize::MAX - counts[token],
                first_seen[token],
            )
        });
        if let Some(max_vocab) = max_vocab {
            candidates.truncate(max_vocab.saturating_sub(2));
        }
        self.vocab = HashMap::from([
            (PAD_TOKEN.to_string(), PAD_ID),
            (UNK_TOKEN.to_string(), UNK_ID),
        ]);
        self.tokens = HashMap::from([
            (PAD_ID, PAD_TOKEN.to_string()),
            (UNK_ID, UNK_TOKEN.to_string()),
        ]);
        for (index, token) in candidates.iter().enumerate() {
            let id = index + 2;
            self.vocab.insert(token.clone(), id);
            self.tokens.insert(id, token.clone());
        }
        self.vocab.clone()
    }

    pub fn encode(
        &self,
        text: &str,
        max_length: Option<usize>,
        pad_to: Option<usize>,
    ) -> Vec<usize> {
        let mut ids: Vec<usize> = self
            .tokenize(text)
            .iter()
            .map(|token| self.vocab.get(token).copied().unwrap_or(UNK_ID))
            .collect();
        if let Some(max_length) = max_length {
            ids.truncate(max_length);
        }
        if let Some(pad_to) = pad_to {
            ids.truncate(pad_to);
            ids.resize(pad_to, PAD_ID);
        }
        ids
    }

    pub fn decode(&self, ids: &[usize], skip_special: bool) -> String {
        let mut parts = String::new();
        for &id in ids {
            let Some(token) = self.tokens.get(&id) else {
                continue;
            };
            if skip_special && (token == PAD_TOKEN || token == UNK_TOKEN) {
                continue;
            }
            let chars: Vec<char> = token.chars().collect();
            if chars.len() == 1 && is_punctuation(chars[0]) {
                parts.push(chars[0]);
            } else {
                if !parts.is_empty() {
                    parts.push(' ');
                }
                parts.push_str(token);
            }
        }
        parts
    }

    pub fn vocab(&self) -> HashMap<String, usize> {
        self.vocab.clone()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trained() -> Tokenizer {
        let mut tok = Tokenizer::new(true);
        tok.build_vocab(
            &["hello world".to_string(), "hello there".to_string()],
            None,
            1,
        );
        tok
    }

    #[test]
    fn tokenize_splits_words_and_punctuation() {
        let tok = Tokenizer::new(true);
        assert_eq!(
            tok.tokenize("Hello, World! How are you?"),
            vec!["hello", ",", "world", "!", "how", "are", "you", "?"]
        );
    }

    #[test]
    fn tokenize_keeps_case_when_disabled() {
        let tok = Tokenizer::new(false);
        assert_eq!(tok.tokenize("Hello World"), vec!["Hello", "World"]);
    }

    #[test]
    fn vocab_orders_by_frequency_then_first_seen() {
        let tok = trained();
        assert_eq!(tok.vocab()["hello"], 2);
        assert_eq!(tok.vocab()["world"], 3);
        assert_eq!(tok.vocab()["there"], 4);
        assert_eq!(tok.vocab_size(), 5);
    }

    #[test]
    fn encode_uses_unk_for_missing_words() {
        let tok = trained();
        assert_eq!(tok.encode("hello nobody", None, None), vec![2, UNK_ID]);
    }

    #[test]
    fn encode_truncates_and_pads() {
        let tok = trained();
        assert_eq!(tok.encode("hello world there", Some(2), None), vec![2, 3]);
        assert_eq!(
            tok.encode("hello", None, Some(3)),
            vec![2, PAD_ID, PAD_ID]
        );
    }

    #[test]
    fn decode_round_trips_and_skips_padding() {
        let tok = trained();
        let ids = tok.encode("hello, world", None, Some(5));
        assert_eq!(tok.decode(&ids, true), "hello, world");
        assert_eq!(tok.decode(&ids, false), "hello, world <pad> <pad>");
    }
}
