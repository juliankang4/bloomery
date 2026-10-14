//! An embedding model: a trunk every position of which attends both ways,
//! then one row per input, the mean of the token rows.

/// What an encoder carries past the trunk its [`crate::ModelSpec`] lists.
/// Its GQA layers have equal key and value widths, no attention sinks and no
/// value scale.
/// Its layers' [`crate::Gqa::window`] is `None`; the window is [`Self::reach`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncoderSpec {
    /// One entry per [`crate::ModelSpec::layers`], in order: the keys a row
    /// attends on each side of its own position; `None`: every key.
    /// A file's `attention.sliding_window` is the window's whole span:
    /// llama.cpp masks a key when
    /// `|i - j| > sliding_window / 2` (`LLAMA_SWA_TYPE_SYMMETRIC`).
    pub reach: Vec<Option<u32>>,
    /// `embedding_length_per_layer_input`: the values of each layer's own
    /// input, projected from the scaled token rows; `None`: no layer has one.
    pub per_layer: Option<u32>,
    /// The widths of the linear maps after the output norm, in order; the last
    /// is the embedding's. With no activation between them they commute with
    /// the mean.
    pub proj: Vec<u32>,
}
