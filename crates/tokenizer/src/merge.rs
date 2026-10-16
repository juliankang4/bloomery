//! Adjacent-symbol merging, shared by rank and score ordered encoders.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::spelling::Spelling;

pub(crate) const NONE: u32 = u32::MAX;

#[derive(Clone, Copy)]
pub(crate) struct Sym {
    /// Bytes this symbol covers; 0 once merged into its left neighbour.
    pub(crate) len: u32,
    pub(crate) iid: u32,
    prev: u32,
    next: u32,
}

/// `(priority, left, right, size at queueing, merged id)`, smallest first.
type Heap<P> = BinaryHeap<Reverse<(P, u32, u32, u32, u32)>>;

pub(crate) struct Scratch<P: Ord> {
    pub(crate) syms: Vec<Sym>,
    heap: Heap<P>,
}

impl<P: Ord> Default for Scratch<P> {
    fn default() -> Self {
        Self {
            syms: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }
}

impl<P: Ord> Scratch<P> {
    /// Start one symbol per byte or UTF-8 character, indexed by its lead byte.
    pub(crate) fn symbols(
        &mut self,
        word: &[u8],
        spelling: Spelling,
        mut id: impl FnMut(&[u8]) -> u32,
    ) {
        let n = word.len();
        assert!(
            u32::try_from(n).is_ok(),
            "a tokenizer word of {n} bytes does not fit the u32 symbol links"
        );
        let at = |i: usize| u32::try_from(i).expect("at most the word length, checked above");
        self.syms.clear();
        self.syms.resize(
            n,
            Sym {
                len: 0,
                iid: NONE,
                prev: NONE,
                next: NONE,
            },
        );
        let (mut i, mut prev) = (0, NONE);
        while i < n {
            let len = match spelling {
                Spelling::ByteLevel => 1,
                Spelling::Metaspace => utf8_len(word[i]).min(n - i),
            };
            self.syms[i] = Sym {
                len: at(len),
                iid: id(&word[i..i + len]),
                prev,
                next: if i + len == n { NONE } else { at(i + len) },
            };
            prev = at(i);
            i += len;
        }
    }

    /// Merge eligible pairs in priority order, breaking ties at the leftmost pair.
    pub(crate) fn run(&mut self, mut pair: impl FnMut(u32, Sym, Sym) -> Option<(P, u32)>) {
        self.heap.clear();
        for i in 0..self.syms.len() {
            let sym = self.syms[i];
            if sym.len > 0 && sym.next != NONE {
                let left = u32::try_from(i).expect("symbols bounds the word length");
                self.push(left, sym.next, &mut pair);
            }
        }
        while let Some(Reverse((_, left, right, size, merged))) = self.heap.pop() {
            let (l, r) = (self.syms[left as usize], self.syms[right as usize]);
            if l.len == 0 || r.len == 0 || l.len + r.len != size {
                continue;
            }
            self.syms[left as usize] = Sym {
                len: size,
                iid: merged,
                next: r.next,
                ..l
            };
            self.syms[right as usize].len = 0;
            if r.next != NONE {
                self.syms[r.next as usize].prev = left;
            }
            if l.prev != NONE {
                self.push(l.prev, left, &mut pair);
            }
            if r.next != NONE {
                self.push(left, r.next, &mut pair);
            }
        }
    }

    fn push(
        &mut self,
        left: u32,
        right: u32,
        pair: &mut impl FnMut(u32, Sym, Sym) -> Option<(P, u32)>,
    ) {
        let (l, r) = (self.syms[left as usize], self.syms[right as usize]);
        if let Some((priority, merged)) = pair(left, l, r) {
            self.heap
                .push(Reverse((priority, left, right, l.len + r.len, merged)));
        }
    }
}

/// `unicode_len_utf8`: continuation bytes count as one byte.
fn utf8_len(lead: u8) -> usize {
    match lead >> 4 {
        0xC | 0xD => 2,
        0xE => 3,
        0xF => 4,
        _ => 1,
    }
}
