//! Python extension module for the ML inference server.
//!
//! Today it exposes only the tokenizer. The server does not call it yet;
//! Day 7 wires it into preprocessing with a fallback when it is missing.
//!
//! Build with maturin (`maturin develop`, from the rust/ directory) or
//! with `cargo build --release` and copy the shared library onto the
//! Python path as `ml_rust`.

use pyo3::prelude::*;
use std::collections::HashMap;

mod tokenizer;
use tokenizer::Tokenizer as CoreTokenizer;

#[pyclass(name = "Tokenizer")]
struct PyTokenizer {
    inner: CoreTokenizer,
}

#[pymethods]
impl PyTokenizer {
    #[new]
    #[pyo3(signature = (lowercase=true))]
    fn new(lowercase: bool) -> Self {
        PyTokenizer { inner: CoreTokenizer::new(lowercase) }
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        self.inner.tokenize(text)
    }

    #[pyo3(signature = (texts, max_vocab=None, min_freq=1))]
    fn build_vocab(
        &mut self,
        texts: Vec<String>,
        max_vocab: Option<usize>,
        min_freq: usize,
    ) -> HashMap<String, usize> {
        self.inner.build_vocab(&texts, max_vocab, min_freq)
    }

    #[pyo3(signature = (text, max_length=None, pad_to=None))]
    fn encode(
        &self,
        text: &str,
        max_length: Option<usize>,
        pad_to: Option<usize>,
    ) -> Vec<usize> {
        self.inner.encode(text, max_length, pad_to)
    }

    #[pyo3(signature = (ids, skip_special=true))]
    fn decode(&self, ids: Vec<usize>, skip_special: bool) -> String {
        self.inner.decode(&ids, skip_special)
    }

    fn vocab(&self) -> HashMap<String, usize> {
        self.inner.vocab()
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

#[pymodule]
fn ml_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    Ok(())
}
