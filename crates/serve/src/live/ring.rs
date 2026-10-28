//! The requests that ended, the newest [`KEPT`] of them: what the live view's
//! Requests tab lists. The engine thread pushes one where it books a request's
//! counters ([`crate::worker`]); the view copies the ring when it draws.

use std::collections::VecDeque;
use std::time::SystemTime;

use crate::genloop::Timings;

/// The requests the ring keeps; an older one leaves as a new one comes.
pub(crate) const KEPT: usize = 64;

/// How a request ended, by the distinctions the server already makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ending {
    /// It finished: its `finish_reason` (`stop` or `length`).
    Done(&'static str),
    /// Its client went away.
    ClientClosed,
    /// The engine failed in its call.
    EngineError,
    /// Any other error ended it.
    Error,
    /// The server's orderly stop ended it.
    Stopped,
}

impl Ending {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Ending::Done(reason) => reason,
            Ending::ClientClosed => "client closed",
            Ending::EngineError => "engine error",
            Ending::Error => "error",
            Ending::Stopped => "stopped",
        }
    }
}

/// One request that ended: when, its timings, and how.
#[derive(Clone, Debug)]
pub(crate) struct Finished {
    pub(crate) at: SystemTime,
    pub(crate) timings: Timings,
    pub(crate) ending: Ending,
}

/// The newest [`KEPT`] requests that ended, oldest first.
#[derive(Default)]
pub(crate) struct Ring(VecDeque<Finished>);

impl Ring {
    /// Keeps `f`, dropping the oldest when the ring is full.
    pub(crate) fn push(&mut self, f: Finished) {
        if self.0.len() == KEPT {
            self.0.pop_front();
        }
        self.0.push_back(f);
    }

    /// A copy of the requests kept, newest first.
    pub(crate) fn newest_first(&self) -> Vec<Finished> {
        self.0.iter().rev().cloned().collect()
    }
}
