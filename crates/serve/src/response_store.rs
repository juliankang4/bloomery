//! The responses `/v1/responses` keeps ([`super::responses`]): each stored
//! response as `GET /v1/responses/{id}` answers it, with the conversation it
//! ends for a later request's `previous_response_id`, and each background
//! response while it runs, with the client its cancel ends.
//!
//! The store is bounded: at most [`MAX_RESPONSES`] responses and
//! [`MAX_BYTES`] bytes of their JSON text (each response's and its
//! conversation's items'). Past either bound the oldest finished response is
//! dropped first; a response past [`MAX_BYTES`] alone is dropped as it is
//! kept. An oversized background result is kept as a named failure instead.
//! A background response is never dropped while it runs: it is admitted
//! only while the running ones with it fit both bounds, its bytes charged at
//! its longest running status or size failure, plus the worker's input items.
//! This bounds serialized JSON, rather than the allocations of its value trees.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use super::{Client, relock};

/// The responses the store holds at most.
const MAX_RESPONSES: usize = 1024;

/// The bytes of JSON text the store holds at most.
const MAX_BYTES: usize = 256 << 20;

/// The queued response with a result that cannot fit the store.
fn oversized(mut response: Value) -> Value {
    response["status"] = json!("failed");
    response["error"] = json!({
        "code": "server_error",
        "message": "background response exceeds the response store byte limit",
    });
    response
}

/// Why a stored response cannot serve a request.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Refusal {
    /// No response of that id is kept: never stored, deleted, or dropped as
    /// the oldest.
    Missing,
    /// The response has no conversation to continue, or cannot be
    /// cancelled, in this status.
    Status(String),
}

/// One kept response.
struct Entry {
    id: String,
    /// The response object, its `status` the current one.
    response: Value,
    /// The input items, a previous response's conversation first, then this
    /// response's output items; empty until it finishes.
    conversation: Vec<Value>,
    /// The running background response's client, which a cancel ends.
    job: Option<Arc<Client>>,
    /// What the entry charges against [`MAX_BYTES`].
    bytes: usize,
}

impl Entry {
    fn status(&self) -> &str {
        self.response["status"].as_str().unwrap_or_default()
    }

    fn charge(&mut self) {
        self.bytes = self.id.len()
            + self.response.to_string().len()
            + self
                .conversation
                .iter()
                .map(|v| v.to_string().len())
                .sum::<usize>();
        if self.job.is_some() {
            // A running response's status grows to `in_progress` or
            // `cancelled` in place; its charge holds the longer one.
            self.bytes += "in_progress".len().saturating_sub(self.status().len());
            self.bytes = self
                .bytes
                .max(self.id.len() + oversized(self.response.clone()).to_string().len());
        }
    }

    /// Ends the running job, if any, as `cancelled`.
    fn cancel(&mut self) {
        if let Some(job) = self.job.take() {
            job.cancel();
            self.response["status"] = json!("cancelled");
        }
    }
}

/// The kept responses, oldest first.
pub(super) struct Store {
    max_responses: usize,
    max_bytes: usize,
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    bytes: usize,
    entries: VecDeque<Entry>,
}

impl Default for Store {
    fn default() -> Self {
        Store::new(MAX_RESPONSES, MAX_BYTES)
    }
}

impl Store {
    fn new(max_responses: usize, max_bytes: usize) -> Self {
        Store {
            max_responses,
            max_bytes,
            inner: Mutex::default(),
        }
    }

    /// Keeps a finished response and its conversation.
    pub(super) fn keep(&self, response: Value, conversation: Vec<Value>) {
        self.push(response, conversation, None, 0);
    }

    /// Keeps a background response that `job` runs, in its first status;
    /// false, and nothing kept, when the running responses with it would
    /// pass either bound.
    pub(super) fn start(&self, response: Value, conversation: &[Value], job: Arc<Client>) -> bool {
        let reserved = conversation.iter().map(|v| v.to_string().len()).sum();
        self.push(response, Vec::new(), Some(job), reserved)
    }

    fn push(
        &self,
        response: Value,
        conversation: Vec<Value>,
        job: Option<Arc<Client>>,
        reserved: usize,
    ) -> bool {
        let mut entry = Entry {
            id: response["id"].as_str().unwrap_or_default().to_owned(),
            response,
            conversation,
            job,
            bytes: 0,
        };
        entry.charge();
        entry.bytes += reserved;
        if entry.bytes > self.max_bytes {
            return false;
        }
        let mut inner = relock(&self.inner);
        if entry.job.is_some() {
            let (count, bytes) = inner
                .entries
                .iter()
                .filter(|e| e.job.is_some())
                .fold((0, 0), |(n, b), e| (n + 1, b + e.bytes));
            if count >= self.max_responses || entry.bytes > self.max_bytes - bytes {
                return false;
            }
        }
        inner.bytes += entry.bytes;
        inner.entries.push_back(entry);
        self.trim(&mut inner);
        true
    }

    /// The running background response `id` in progress: its `status` from
    /// `queued` to `in_progress`.
    pub(super) fn progress(&self, id: &str) {
        let mut inner = relock(&self.inner);
        if let Some(e) = inner.entries.iter_mut().find(|e| e.id == id)
            && e.job.is_some()
        {
            e.response["status"] = json!("in_progress");
        }
    }

    /// The running background response `id` finished as `response` with its
    /// conversation; a response cancelled or deleted meanwhile stays as it is.
    pub(super) fn finish(&self, id: &str, response: Value, conversation: Vec<Value>) {
        let mut inner = relock(&self.inner);
        let Some(at) = inner
            .entries
            .iter()
            .position(|e| e.id == id && e.job.is_some())
        else {
            return;
        };
        let mut e = inner.entries.remove(at).expect("the entry was found");
        inner.bytes -= e.bytes;
        e.job = None;
        let pending = std::mem::replace(&mut e.response, response);
        e.conversation = conversation;
        e.charge();
        if e.bytes > self.max_bytes {
            e.response = oversized(pending);
            e.conversation.clear();
            e.charge();
        }
        inner.bytes += e.bytes;
        inner.entries.insert(at, e);
        self.trim(&mut inner);
    }

    /// The response `id` as it stands.
    pub(super) fn get(&self, id: &str) -> Option<Value> {
        let inner = relock(&self.inner);
        let e = inner.entries.iter().find(|e| e.id == id)?;
        Some(e.response.clone())
    }

    /// The conversation a request continues with `previous_response_id`
    /// `id`: only a `completed` or `incomplete` response has one.
    pub(super) fn conversation(&self, id: &str) -> Result<Vec<Value>, Refusal> {
        let inner = relock(&self.inner);
        let e = inner
            .entries
            .iter()
            .find(|e| e.id == id)
            .ok_or(Refusal::Missing)?;
        match e.status() {
            "completed" | "incomplete" => Ok(e.conversation.clone()),
            other => Err(Refusal::Status(other.to_owned())),
        }
    }

    /// Cancels the running background response `id` and answers it
    /// `cancelled`; a cancelled one answers as it is, again.
    pub(super) fn cancel(&self, id: &str) -> Result<Value, Refusal> {
        let mut inner = relock(&self.inner);
        let e = inner
            .entries
            .iter_mut()
            .find(|e| e.id == id)
            .ok_or(Refusal::Missing)?;
        e.cancel();
        match e.status() {
            "cancelled" => Ok(e.response.clone()),
            other => Err(Refusal::Status(other.to_owned())),
        }
    }

    /// Drops the response `id`, cancelling it first when it runs; whether it
    /// was kept.
    pub(super) fn delete(&self, id: &str) -> bool {
        let mut inner = relock(&self.inner);
        let Some(at) = inner.entries.iter().position(|e| e.id == id) else {
            return false;
        };
        let mut e = inner.entries.remove(at).expect("the entry was found");
        e.cancel();
        inner.bytes -= e.bytes;
        true
    }

    /// Drops the oldest finished responses until both bounds hold, or only
    /// running ones are left.
    fn trim(&self, inner: &mut Inner) {
        while inner.entries.len() > self.max_responses || inner.bytes > self.max_bytes {
            let Some(at) = inner.entries.iter().position(|e| e.job.is_none()) else {
                return;
            };
            let e = inner.entries.remove(at).expect("the entry was found");
            inner.bytes -= e.bytes;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::{Value, json};

    use super::{Client, Refusal, Store};

    fn response(id: &str, status: &str) -> Value {
        json!({ "id": id, "object": "response", "status": status })
    }

    fn ids(store: &Store) -> Vec<String> {
        super::relock(&store.inner)
            .entries
            .iter()
            .map(|e| e.id.clone())
            .collect()
    }

    #[test]
    fn the_oldest_finished_response_goes_first_past_either_bound() {
        let store = Store::new(2, usize::MAX);
        for id in ["a", "b", "c"] {
            store.keep(response(id, "completed"), vec![json!({"role": "user"})]);
        }
        assert_eq!(ids(&store), ["b", "c"]);
        assert_eq!(store.get("a"), None);
        assert_eq!(store.conversation("a"), Err(Refusal::Missing));

        // Three entries' bytes hold two of these.
        let one = {
            let s = Store::new(usize::MAX, usize::MAX);
            s.keep(response("x", "completed"), vec![json!("12345678")]);
            super::relock(&s.inner).bytes
        };
        let store = Store::new(usize::MAX, 2 * one + one / 2);
        for id in ["d", "e", "f"] {
            store.keep(response(id, "completed"), vec![json!("12345678")]);
        }
        assert_eq!(ids(&store), ["e", "f"]);
        assert_eq!(super::relock(&store.inner).bytes, 2 * one);
        // A response past the byte bound alone is not kept.
        store.keep(response("g", "completed"), vec![json!("x".repeat(3 * one))]);
        assert_eq!(ids(&store), ["e", "f"]);
        assert!(store.delete("e"));
        assert!(!store.delete("e"));
        assert_eq!(super::relock(&store.inner).bytes, one);
    }

    #[test]
    fn a_running_background_response_is_never_dropped_and_cancels_once() {
        let store = Store::new(1, usize::MAX);
        let job = Arc::new(Client::detached());
        assert!(store.start(response("bg", "queued"), &[], Arc::clone(&job)));
        store.keep(response("a", "completed"), Vec::new());
        assert_eq!(ids(&store), ["bg"]);
        store.progress("bg");
        assert_eq!(store.get("bg").expect("kept")["status"], "in_progress");
        assert_eq!(
            store.conversation("bg"),
            Err(Refusal::Status("in_progress".into()))
        );
        assert!(job.check().is_ok());
        assert_eq!(
            store.cancel("bg").expect("cancelled")["status"],
            "cancelled"
        );
        assert!(job.check().is_err(), "the cancel ends the job's client");
        assert_eq!(store.cancel("bg").expect("again")["status"], "cancelled");
        // The job's late end does not overwrite the cancel.
        store.finish("bg", response("bg", "completed"), Vec::new());
        assert_eq!(store.get("bg").expect("kept")["status"], "cancelled");
        store.keep(response("b", "completed"), Vec::new());
        assert_eq!(ids(&store), ["b"]);
        assert_eq!(store.cancel("b"), Err(Refusal::Status("completed".into())));

        let job = Arc::new(Client::detached());
        assert!(store.start(response("bg", "queued"), &[], Arc::clone(&job)));
        store.progress("bg");
        assert!(job.check().is_ok());
        assert!(store.delete("bg"));
        assert!(job.check().is_err(), "delete ends the running job's client");
        store.finish("bg", response("bg", "completed"), Vec::new());
        assert_eq!(store.get("bg"), None, "a late finish cannot undo delete");
        assert_eq!(super::relock(&store.inner).bytes, 0);
    }

    #[test]
    fn running_responses_are_admitted_only_within_both_bounds() {
        let charge = |s: &Store| super::relock(&s.inner).bytes;
        let running = |id: &str| response(id, "queued");
        let one = {
            let s = Store::new(usize::MAX, usize::MAX);
            assert!(s.start(running("a"), &[], Arc::new(Client::detached())));
            charge(&s)
        };
        // Its charge holds its longest running status: progress and cancel
        // keep the stored JSON within it.
        let store = Store::new(usize::MAX, one);
        assert!(store.start(running("a"), &[], Arc::new(Client::detached())));
        store.progress("a");
        let size = |s: &Store| {
            let inner = super::relock(&s.inner);
            inner
                .entries
                .iter()
                .map(|e| e.id.len() + e.response.to_string().len())
                .sum::<usize>()
        };
        assert!(size(&store) <= charge(&store));
        store.cancel("a").expect("cancelled");
        assert!(size(&store) <= charge(&store));

        // Running responses fill the bytes: the next is refused and none is
        // dropped; a finished one makes no room for it.
        let store = Store::new(usize::MAX, 2 * one + one / 2);
        assert!(store.start(running("a"), &[], Arc::new(Client::detached())));
        assert!(store.start(running("b"), &[], Arc::new(Client::detached())));
        assert!(!store.start(running("c"), &[], Arc::new(Client::detached())));
        assert_eq!(ids(&store), ["a", "b"]);
        assert_eq!(charge(&store), 2 * one);
        // And the count.
        let store = Store::new(2, usize::MAX);
        assert!(store.start(running("a"), &[], Arc::new(Client::detached())));
        store.keep(response("k", "completed"), Vec::new());
        assert!(store.start(running("b"), &[], Arc::new(Client::detached())));
        assert!(!store.start(running("c"), &[], Arc::new(Client::detached())));
        assert_eq!(ids(&store), ["a", "b"]);
        // A finished one frees its place.
        store.finish("a", response("a", "completed"), Vec::new());
        assert!(store.start(running("c"), &[], Arc::new(Client::detached())));
    }

    #[test]
    fn pending_history_is_charged_before_a_worker_is_admitted() {
        let input = vec![json!({"role": "user", "content": "hi", "metadata": "x".repeat(512)})];
        let base = {
            let s = Store::new(usize::MAX, usize::MAX);
            assert!(s.start(response("a", "queued"), &[], Arc::new(Client::detached())));
            super::relock(&s.inner).bytes
        };
        let budget = base + input[0].to_string().len();
        let store = Store::new(usize::MAX, budget);
        assert!(store.start(
            response("a", "queued"),
            &input,
            Arc::new(Client::detached())
        ));
        assert_eq!(super::relock(&store.inner).bytes, budget);
        assert!(!store.start(response("b", "queued"), &[], Arc::new(Client::detached())));
        assert_eq!(ids(&store), ["a"]);
        store.finish("a", response("a", "completed"), Vec::new());
        assert!(store.start(response("b", "queued"), &[], Arc::new(Client::detached())));

        let store = Store::new(usize::MAX, budget - 1);
        assert!(!store.start(
            response("a", "queued"),
            &input,
            Arc::new(Client::detached())
        ));
        assert!(ids(&store).is_empty());
        assert_eq!(super::relock(&store.inner).bytes, 0);
    }

    #[test]
    fn an_oversized_background_result_is_a_named_failure() {
        let budget = {
            let s = Store::new(usize::MAX, usize::MAX);
            assert!(s.start(response("bg", "queued"), &[], Arc::new(Client::detached())));
            super::relock(&s.inner).bytes
        };
        for history in [false, true] {
            let store = Store::new(usize::MAX, budget);
            assert!(store.start(response("bg", "queued"), &[], Arc::new(Client::detached())));
            store.progress("bg");
            let mut result = response("bg", "completed");
            let input = if history {
                vec![json!("x".repeat(budget))]
            } else {
                result["output"] = json!(["x".repeat(budget)]);
                Vec::new()
            };
            store.finish("bg", result, input);
            let failed = store.get("bg").expect("the result remains pollable");
            assert_eq!(failed["status"], "failed");
            assert_eq!(failed["error"]["code"], "server_error");
            assert_eq!(
                failed["error"]["message"],
                "background response exceeds the response store byte limit"
            );
            assert_eq!(
                store.conversation("bg"),
                Err(Refusal::Status("failed".into()))
            );
            let inner = super::relock(&store.inner);
            assert!(inner.bytes <= budget);
            assert!(inner.entries[0].job.is_none());
            assert!(inner.entries[0].conversation.is_empty());
        }
    }
}
