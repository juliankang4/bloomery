//! The buffers of the common head128 projection/attention scratch.

use super::{SiteAct, try_q8_act_bytes};

/// Quantized wide forms beside the caller-owned f32 input rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Forms {
    pub q128: bool,
    pub q32: bool,
}

impl Forms {
    /// Include the activation form a selected projection reads.
    #[must_use]
    pub const fn with(self, form: SiteAct) -> Self {
        Self {
            q128: self.q128 || matches!(form, SiteAct::Q8x128),
            q32: self.q32 || matches!(form, SiteAct::Q8x32 | SiteAct::Q8Blocks32),
        }
    }

    /// Bytes of these forms over `rows` columns of `width` values.
    #[must_use]
    pub fn bytes(self, rows: u64, width: u64) -> Option<u64> {
        let q128 = if self.q128 {
            try_q8_act_bytes(rows, width)?
        } else {
            0
        };
        let q32 = if self.q32 {
            wide32_bytes(rows, width)?
        } else {
            0
        };
        q128.checked_add(q32)
    }
}

/// Bytes of wide q8 blocks of 32: twenty words per padded 64-value step.
#[must_use]
pub const fn wide32_bytes(rows: u64, width: u64) -> Option<u64> {
    match rows.checked_mul(width.div_ceil(64)) {
        Some(steps) => steps.checked_mul(80),
        None => None,
    }
}

/// The loaded projection widths, partial segments and actual selected forms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Head128Scratch {
    pub hidden: u64,
    pub query: u64,
    pub key_value: u64,
    pub heads: u64,
    pub segments: u64,
    pub input: Forms,
    pub output: Forms,
}

impl Head128Scratch {
    /// Checked bytes for Q/K/V, normalized/residual rows, attention/partials,
    /// selected wide forms and the one-matrix route table.
    #[must_use]
    pub fn bytes(self, rows: u64) -> Option<u64> {
        let partial = self.heads.checked_mul(self.segments)?.checked_mul(130)?;
        let values = self
            .hidden
            .checked_mul(2)?
            .checked_add(self.query.checked_mul(2)?)?
            .checked_add(self.key_value.checked_mul(2)?)?
            .checked_add(partial)?;
        let floats = values.checked_mul(rows)?.checked_mul(4)?;
        let route = super::ffn::dense_route_bytes(rows)?;
        floats
            .checked_add(self.input.bytes(rows, self.hidden)?)?
            .checked_add(self.output.bytes(rows, self.query)?)?
            .checked_add(route)
    }
}
