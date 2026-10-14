//! The plan's side of an image input, free of the encoder crates so a placement module every
//! build includes can take it: what a plan adds per position of a sequence for a media load
//! ([`table_bytes_per_position`]). The tower's reserve on the stage card is a [`crate::beside`]
//! reserve. The seat that owns the rest of the vision seat (`crate::vision_seat`) re-exports it.

/// The bytes a plan adds per position of a sequence for a media load: the sequence's own rotation
/// table (`gpu::media::SeqRope`) holds a row of `width` f32 for each of its positions, `width`
/// being the f32 a rope row turns (64 for the Qwen3-VL family's text, 256 B a position).
#[must_use]
pub const fn table_bytes_per_position(width: usize) -> u64 {
    width as u64 * size_of::<f32>() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table term: a row of the rope's width in f32 a position (256 B at 64).
    #[test]
    fn the_table_term_is_a_row_of_f32_a_position() {
        assert_eq!(table_bytes_per_position(64), 256);
        assert_eq!(table_bytes_per_position(128), 512);
        assert_eq!(table_bytes_per_position(0), 0);
    }
}
