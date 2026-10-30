//! The byte form a host state is written in and read back from, what a slot
//! file carries after the server's header: little-endian values through one
//! buffer of [`CHUNK`] bytes each way, a count a u64, a name a u32 length
//! and its bytes. Each owner of a state lays out its own parts with
//! [`Put`] and reads them back with [`Take`] in the same order
//! ([`super::saved::SeqState::put`], the DeepSeek V4.1 body's snapshot).
//!
//! A reader takes its counts from the stream it reads, so no count sizes an
//! allocation before the bytes it covers have arrived: [`Take::vals`] grows
//! its array a chunk at a time, and a stream that ends early is refused by
//! name with the part it ends in.

use std::fmt;
use std::io::{self, Read, Write};

use crate::GpuError;

const WHAT: &str = "state bytes";

/// Bytes of the write and read buffer.
const CHUNK: usize = 1 << 16;

/// The longest name the form carries: a path's limit.
const NAME_MAX: usize = 4096;

/// Why a write or a read of the byte form failed.
#[derive(Debug)]
pub enum Fail {
    /// Refused by name ([`GpuError::Shape`]): a stream that is not a state
    /// this reader takes (another tag, format, model or layout, a short
    /// stream, trailing bytes, a count out of range), or a state that cannot
    /// be written as it stands. Nothing on the device was touched.
    Refused(GpuError),
    /// The stream itself failed: a write or a read it refused (a closed
    /// pipe, a full disk), neither the request's form nor the model's.
    Io(io::Error),
}

impl fmt::Display for Fail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fail::Refused(e) => write!(f, "refused: {e}"),
            Fail::Io(e) => write!(f, "the stream failed: {e}"),
        }
    }
}

impl std::error::Error for Fail {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Fail::Refused(e) => Some(e),
            Fail::Io(e) => Some(e),
        }
    }
}

/// A refusal by name of the byte form ([`Fail::Refused`]).
#[must_use]
pub fn refused(detail: impl Into<String>) -> Fail {
    Fail::Refused(GpuError::shape(WHAT, detail))
}

/// A value the byte form carries, little-endian.
pub trait Le: Copy {
    const SIZE: usize;
    fn put(self, out: &mut Vec<u8>);
    /// The value of `b`, [`Le::SIZE`] bytes.
    fn get(b: &[u8]) -> Self;
}

macro_rules! le {
    ($($t:ty),*) => {$(
        impl Le for $t {
            const SIZE: usize = size_of::<$t>();
            fn put(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
            fn get(b: &[u8]) -> $t {
                let mut a = [0u8; size_of::<$t>()];
                a.copy_from_slice(b);
                <$t>::from_le_bytes(a)
            }
        }
    )*};
}

le!(u8, u16, u32, u64, f32);

/// The writer: `out` through a buffer of at most [`CHUNK`] bytes, the bytes
/// written counted. An error of `out` is the stream's ([`Fail::Io`]).
pub struct Put<'a> {
    out: &'a mut dyn Write,
    buf: Vec<u8>,
    bytes: u64,
}

impl<'a> Put<'a> {
    pub fn new(out: &'a mut dyn Write) -> Put<'a> {
        Put {
            out,
            buf: Vec::with_capacity(CHUNK),
            bytes: 0,
        }
    }

    /// The bytes written to `out` so far: every one once [`Put::flush`] ran.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn one<T: Le>(&mut self, v: T) -> Result<(), Fail> {
        self.vals(&[v])
    }

    pub fn vals<T: Le>(&mut self, vals: &[T]) -> Result<(), Fail> {
        for c in vals.chunks(CHUNK / T::SIZE) {
            if self.buf.len() + c.len() * T::SIZE > CHUNK {
                self.flush()?;
            }
            for &v in c {
                v.put(&mut self.buf);
            }
        }
        Ok(())
    }

    pub fn count(&mut self, v: usize) -> Result<(), Fail> {
        self.one(v as u64)
    }

    /// `vals`' count, then its values.
    pub fn array<T: Le>(&mut self, vals: &[T]) -> Result<(), Fail> {
        self.count(vals.len())?;
        self.vals(vals)
    }

    /// `b` as a name: its length (u32), then its bytes; refused by name past
    /// [`NAME_MAX`].
    pub fn name(&mut self, b: &[u8]) -> Result<(), Fail> {
        let len = u32::try_from(b.len())
            .ok()
            .filter(|&l| l as usize <= NAME_MAX)
            .ok_or_else(|| {
                refused(format!(
                    "a name of {} bytes, past the {NAME_MAX} the form carries",
                    b.len()
                ))
            })?;
        self.one(len)?;
        self.vals(b)
    }

    /// `name`, one of `names`, as one past its place there (u8), none as 0;
    /// refused by name for a name `names` does not hold.
    pub fn named(
        &mut self,
        name: Option<&str>,
        names: &[&'static str],
        part: &str,
    ) -> Result<(), Fail> {
        let code = match name {
            None => 0,
            Some(n) => names
                .iter()
                .position(|&m| m == n)
                .and_then(|i| u8::try_from(i + 1).ok())
                .ok_or_else(|| refused(format!("{part} the form names no code for: {n}")))?,
        };
        self.one(code)
    }

    pub fn flush(&mut self) -> Result<(), Fail> {
        self.out.write_all(&self.buf).map_err(Fail::Io)?;
        self.bytes += self.buf.len() as u64;
        self.buf.clear();
        Ok(())
    }
}

/// The reader: `input` through a buffer of at most [`CHUNK`] bytes, the
/// bytes read counted. A stream that ends early is refused by name, with the
/// part it ends in; any other error of `input` is the stream's
/// ([`Fail::Io`]).
pub struct Take<'a> {
    input: &'a mut dyn Read,
    buf: Vec<u8>,
    bytes: u64,
}

impl<'a> Take<'a> {
    pub fn new(input: &'a mut dyn Read) -> Take<'a> {
        Take {
            input,
            buf: Vec::with_capacity(CHUNK),
            bytes: 0,
        }
    }

    /// The bytes read from `input` so far.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The next `len` bytes, `len` at most [`CHUNK`].
    fn exact(&mut self, len: usize, part: &str) -> Result<&[u8], Fail> {
        self.buf.resize(len, 0);
        self.input
            .read_exact(&mut self.buf)
            .map_err(|e| match e.kind() {
                io::ErrorKind::UnexpectedEof => refused(format!(
                    "a short stream: it ends inside {part}, past byte {}",
                    self.bytes
                )),
                _ => Fail::Io(e),
            })?;
        self.bytes += len as u64;
        Ok(&self.buf)
    }

    pub fn one<T: Le>(&mut self, part: &str) -> Result<T, Fail> {
        Ok(T::get(self.exact(T::SIZE, part)?))
    }

    pub fn count(&mut self, part: &str) -> Result<usize, Fail> {
        let v: u64 = self.one(part)?;
        usize::try_from(v).map_err(|_| refused(format!("{part}: {v} passes usize")))
    }

    /// `len` values, the array grown a chunk at a time as they arrive.
    pub fn vals<T: Le>(&mut self, len: usize, part: &str) -> Result<Vec<T>, Fail> {
        let mut v = Vec::new();
        let mut left = len;
        while left > 0 {
            let n = left.min(CHUNK / T::SIZE);
            let b = self.exact(n * T::SIZE, part)?;
            v.try_reserve(n)
                .map_err(|e| refused(format!("{part}: {len} values of host memory: {e}")))?;
            v.extend(b.chunks_exact(T::SIZE).map(T::get));
            left -= n;
        }
        Ok(v)
    }

    /// [`Put::array`]'s count and values.
    pub fn array<T: Le>(&mut self, part: &str) -> Result<Vec<T>, Fail> {
        let n = self.count(part)?;
        self.vals(n, part)
    }

    pub fn name(&mut self, part: &str) -> Result<Vec<u8>, Fail> {
        let len = self.one::<u32>(part)? as usize;
        if len > NAME_MAX {
            return Err(refused(format!(
                "{part} of {len} bytes, past the {NAME_MAX} the form carries"
            )));
        }
        Ok(self.exact(len, part)?.to_vec())
    }

    /// What [`Put::named`] wrote of `names`; refused by name for a code past
    /// them.
    pub fn named(
        &mut self,
        names: &[&'static str],
        part: &str,
    ) -> Result<Option<&'static str>, Fail> {
        match self.one::<u8>(part)? {
            0 => Ok(None),
            c => names
                .get(usize::from(c) - 1)
                .copied()
                .map(Some)
                .ok_or_else(|| refused(format!("{part} of code {c}"))),
        }
    }

    /// Refused by name unless the stream ends here.
    pub fn end(&mut self) -> Result<(), Fail> {
        let mut b = [0u8; 1];
        loop {
            match self.input.read(&mut b) {
                Ok(0) => return Ok(()),
                Ok(_) => {
                    return Err(refused(format!(
                        "trailing bytes past the state's {}: the form names none",
                        self.bytes
                    )));
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(Fail::Io(e)),
            }
        }
    }
}
