//! Python extension module for the ML inference server.
//!
//! It exposes the tokenizer and batch preparation. The server calls it
//! through app/backend.py, which falls back to pure Python when this
//! extension is missing.
//!
//! Build with maturin (`maturin develop`, from the rust/ directory) or
//! with `cargo build --release` and copy the shared library onto the
//! Python path as `ml_rust`.

use pyo3::prelude::*;
use std::collections::HashMap;

mod preprocessing;
mod tokenizer;
use preprocessing::PreparedBatch as CoreBatch;
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

    #[pyo3(signature = (texts, max_length=None))]
    fn prepare_batch(
        &self,
        texts: Vec<String>,
        max_length: Option<usize>,
    ) -> PyPreparedBatch {
        PyPreparedBatch {
            inner: preprocessing::prepare_batch(&self.inner, &texts, max_length),
        }
    }
}

#[pyclass(name = "PreparedBatch")]
struct PyPreparedBatch {
    inner: CoreBatch,
}

#[pymethods]
impl PyPreparedBatch {
    #[getter]
    fn ids(&self) -> Vec<Vec<usize>> {
        self.inner.ids.clone()
    }

    #[getter]
    fn attention_mask(&self) -> Vec<Vec<u8>> {
        self.inner.attention_mask.clone()
    }

    #[getter]
    fn lengths(&self) -> Vec<usize> {
        self.inner.lengths.clone()
    }
}

#[pymodule]
fn ml_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    m.add_class::<PyPreparedBatch>()?;
    Ok(())
}
