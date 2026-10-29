//! Batch preparation: encode texts, then pad them into a rectangle.
//!
//! This is the same step as `Preprocessor` in app/preprocessing.py, kept
//! in plain Rust with no Python types so it can be unit-tested with
//! cargo test alone. Doing the whole batch in one call matters: a
//! Python loop that calls back into Rust once per text pays the
//! boundary cost on every text, while this pays it once per batch.

use crate::tokenizer::{Tokenizer, PAD_ID};

pub struct PreparedBatch {
    pub ids: Vec<Vec<usize>>,
    pub attention_mask: Vec<Vec<u8>>,
    pub lengths: Vec<usize>,
}

pub fn prepare_batch(
    tokenizer: &Tokenizer,
    texts: &[String],
    max_length: Option<usize>,
) -> PreparedBatch {
    let encoded: Vec<Vec<usize>> = texts
        .iter()
        .map(|text| tokenizer.encode(text, max_length, None))
        .collect();
    let width = encoded.iter().map(|row| row.len()).max().unwrap_or(0);
    let lengths: Vec<usize> = encoded.iter().map(|row| row.len()).collect();
    let mut ids = Vec::with_capacity(encoded.len());
    let mut attention_mask = Vec::with_capacity(encoded.len());
    for row in encoded {
        let mut mask = vec![1u8; row.len()];
        mask.resize(width, 0);
        let mut padded = row;
        padded.resize(width, PAD_ID);
        ids.push(padded);
        attention_mask.push(mask);
    }
    PreparedBatch { ids, attention_mask, lengths }
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
    fn pads_to_longest_and_builds_mask() {
        let tok = trained();
        let batch = prepare_batch(
            &tok,
            &["hello".to_string(), "hello world".to_string()],
            None,
        );
        assert_eq!(batch.ids, vec![vec![2, PAD_ID], vec![2, 3]]);
        assert_eq!(batch.attention_mask, vec![vec![1, 0], vec![1, 1]]);
        assert_eq!(batch.lengths, vec![1, 2]);
    }

    #[test]
    fn truncates_before_padding() {
        let tok = trained();
        let batch = prepare_batch(
            &tok,
            &["hello world there".to_string()],
            Some(2),
        );
        assert_eq!(batch.ids, vec![vec![2, 3]]);
        assert_eq!(batch.lengths, vec![2]);
    }

    #[test]
    fn empty_batch_is_empty() {
        let tok = trained();
        let batch = prepare_batch(&tok, &[], None);
        assert!(batch.ids.is_empty());
        assert!(batch.attention_mask.is_empty());
        assert!(batch.lengths.is_empty());
    }

    #[test]
    fn empty_text_pads_to_batch_width() {
        let tok = trained();
        let batch = prepare_batch(
            &tok,
            &["".to_string(), "hello".to_string()],
            None,
        );
        assert_eq!(batch.ids, vec![vec![PAD_ID], vec![2]]);
        assert_eq!(batch.attention_mask, vec![vec![0], vec![1]]);
        assert_eq!(batch.lengths, vec![0, 1]);
    }
}
