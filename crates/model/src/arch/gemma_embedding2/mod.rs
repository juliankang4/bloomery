//! `gemma-embedding2`: EmbeddingGemma 2, a text encoder. Every layer is
//! grouped-query attention in both directions (a symmetric window on the
//! sliding layers) with a gainless norm on the values, a GELU GLU, norms
//! after both blocks, a per-layer input projected from the token rows and a
//! trained scalar on the layer's output; after the output norm one
//! projection, then the mean of the rows. The file's keys (`hparams`), every
//! tensor's role (`roles`), the tensor names (`names`) and the typed
//! description (`spec`). No program runs it. The dialect's authority is
//! llama.cpp's `src/models/gemma-embedding2.cpp`.

pub mod hparams;
pub mod names;
pub mod roles;
pub mod spec;
