//! One byte budget and LRU order for a side model's pooled vectors and image rows.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use sha2::{Digest, Sha256};

use crate::media::MediaFeed;

/// The server and its side model share this cache for their lifetime.
pub type Shared = Arc<Mutex<Cache>>;

/// A cached result before response normalization, or the tower's BF16 rows.
#[derive(Clone)]
pub enum Value {
    Vector(Vec<f32>),
    Image(Vec<u16>),
}

/// Retained values and entry metadata are charged; container slack and allocator overhead are not.
pub struct Cache {
    limit: usize,
    bytes: usize,
    entries: VecDeque<([u8; 32], Value)>,
}

impl Cache {
    /// Zero disables caching without allocating an owner.
    #[must_use]
    pub fn shared(limit: usize) -> Option<Shared> {
        (limit > 0).then(|| {
            Arc::new(Mutex::new(Self {
                limit,
                bytes: 0,
                entries: VecDeque::new(),
            }))
        })
    }

    /// A hit becomes the most recently used entry.
    pub fn get(&mut self, key: &[u8; 32]) -> Option<Value> {
        let at = self.entries.iter().position(|(k, _)| k == key)?;
        let entry = self.entries.remove(at).expect("the entry was found");
        let value = entry.1.clone();
        self.entries.push_back(entry);
        Some(value)
    }

    /// Oversized results are not retained and do not evict useful entries.
    pub fn insert(&mut self, key: [u8; 32], value: Value) {
        let bytes = Self::charge(&value);
        if bytes > self.limit {
            return;
        }
        if let Some(at) = self.entries.iter().position(|(k, _)| *k == key) {
            let (_, old) = self.entries.remove(at).expect("the entry was found");
            self.bytes -= Self::charge(&old);
        }
        while self.bytes > self.limit - bytes {
            let (_, old) = self.entries.pop_front().expect("charged entries exist");
            self.bytes -= Self::charge(&old);
        }
        self.bytes += bytes;
        self.entries.push_back((key, value));
    }

    fn charge(value: &Value) -> usize {
        size_of::<([u8; 32], Value)>()
            + match value {
                Value::Vector(v) => v.capacity() * size_of::<f32>(),
                Value::Image(v) => v.capacity() * size_of::<u16>(),
            }
    }
}

/// A file and its prepared grid identify a tower input independently of its text position.
/// Non-grid inputs are not retained by this cache.
#[must_use]
pub fn image_key(image: &MediaFeed) -> Option<[u8; 32]> {
    let vision::EncoderInput::Grid(grid) = &image.prepared.input else {
        return None;
    };
    let mut hash = Sha256::new();
    hash.update(b"image");
    hash.update(image.key.0);
    for n in [
        image.prepared.span_len,
        grid.n_vit_h,
        grid.n_vit_w,
        grid.patch_len,
        grid.plan.span_grid().0,
        grid.plan.span_grid().1,
        grid.plan.best_h,
        grid.plan.best_w,
    ] {
        hash.update(n.to_le_bytes());
    }
    Some(hash.finalize().into())
}

/// The pooled result depends on every token and each whole image at its position.
#[must_use]
pub fn vector_key(ids: &[u32], images: &[MediaFeed]) -> Option<[u8; 32]> {
    let mut hash = Sha256::new();
    hash.update(b"vector");
    hash.update(ids.len().to_le_bytes());
    for id in ids {
        hash.update(id.to_le_bytes());
    }
    for image in images {
        hash.update(image.at.to_le_bytes());
        hash.update(image_key(image)?);
    }
    Some(hash.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_include_image_identity_shape_and_position_but_not_tower_text() {
        let image = |key, at, span_len| MediaFeed {
            key: crate::media::ImageKey([key; 32]),
            at,
            prepared: Arc::new(vision::Prepared::of_grid(
                span_len,
                vision::Patches {
                    plan: vision::GridPlan {
                        n_llm_h: 1,
                        n_llm_w: span_len,
                        best_h: 16,
                        best_w: span_len * 16,
                    },
                    n_vit_h: 1,
                    n_vit_w: span_len,
                    patch_len: 1,
                    bf16: vec![0; span_len],
                },
            )),
        };
        let a = image(1, 2, 4);
        let shifted = image(1, 3, 4);
        assert_eq!(image_key(&a), image_key(&shifted));
        assert_ne!(image_key(&a), image_key(&image(2, 2, 4)));
        assert_ne!(image_key(&a), image_key(&image(1, 2, 8)));
        assert_ne!(
            vector_key(&[1, 2], std::slice::from_ref(&a)),
            vector_key(&[1, 2], &[shifted])
        );
        assert_ne!(
            vector_key(&[1, 2], std::slice::from_ref(&a)),
            vector_key(&[2, 1], &[a])
        );
    }

    #[test]
    fn one_budget_evicts_the_least_recent_vector_or_image() {
        assert!(Cache::shared(0).is_none());
        let v = || Value::Vector(vec![1.0, -0.0]);
        let bytes = Cache::charge(&v());
        let shared = Cache::shared(2 * bytes).unwrap();
        let mut cache = shared.lock().unwrap();
        cache.insert([1; 32], v());
        cache.insert([2; 32], Value::Image(vec![1, 2, 3, 4]));
        let Some(Value::Vector(hit)) = cache.get(&[1; 32]) else {
            panic!("vector hit")
        };
        assert_eq!(
            hit.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            vec![1.0f32.to_bits(), (-0.0f32).to_bits()]
        );
        cache.insert([3; 32], v());
        assert!(cache.get(&[2; 32]).is_none());
        cache.insert([4; 32], Value::Image(vec![0; 4 * bytes]));
        assert!(cache.get(&[4; 32]).is_none());
        assert!(cache.get(&[3; 32]).is_some());
        assert!(cache.bytes <= cache.limit);
        assert_ne!(vector_key(&[1, 2], &[]), vector_key(&[2, 1], &[]));
        assert_ne!(vector_key(&[1], &[]), vector_key(&[1, 0], &[]));
    }
}
