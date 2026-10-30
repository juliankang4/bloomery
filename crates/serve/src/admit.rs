//! A key's own limits on the requests it makes, set on its key file line
//! ([`crate::flag::ApiKeys`]): `inflight=N`, the most it may have answered at
//! once, and `rate=N/s` or `rate=N/m` with `burst=B`, a bucket of `B` requests
//! (`N` unless set) that refills `N` a second or a minute, each request taking
//! one. A request past either gets a 429 with `Retry-After` and starts nothing;
//! a key with neither set is limited by nothing. Each key counts apart: one
//! key's requests never use another's room. A call on a stored response takes
//! no in-flight place, so it can cancel the generation that holds one.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::api::relock;

/// The most `N` and `B` a limit field takes, so the bucket's span cannot overflow.
const MOST: u32 = 1_000_000;

/// A key's limits, as its key file line sets them; the default limits nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Limits {
    /// `inflight=N`: the most requests the key may have answered at once.
    inflight: Option<u32>,
    /// `rate=` and `burst=`: how often the key may start a request.
    rate: Option<Rate>,
}

/// A request bucket: one request every `interval` on average, `burst` at once
/// after a quiet spell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Rate {
    /// The `N/s` or `N/m` as written, for the 429's message.
    count: u32,
    per_minute: bool,
    interval: Duration,
    burst: u32,
}

impl Limits {
    /// The limits a key file line's `name=value` fields set. A name that is
    /// not a limit, or a value out of its range, is refused naming the field;
    /// the key itself is never in the message.
    pub(crate) fn parse<'a>(
        fields: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<Limits, String> {
        let mut limits = Limits::default();
        let mut burst = None;
        for (name, value) in fields {
            match name {
                "inflight" => limits.inflight = Some(count(name, value, value)?),
                "burst" => burst = Some(count(name, value, value)?),
                "rate" => {
                    let (n, unit) = value.split_once('/').ok_or_else(|| {
                        format!("rate={value}: takes N/s or N/m, a count a second or a minute")
                    })?;
                    let per_minute = match unit {
                        "s" => false,
                        "m" => true,
                        _ => return Err(format!("rate={value}: the unit is s or m")),
                    };
                    let count = count(name, value, n)?;
                    let unit = Duration::from_secs(if per_minute { 60 } else { 1 });
                    limits.rate = Some(Rate {
                        count,
                        per_minute,
                        interval: unit / count,
                        burst: count,
                    });
                }
                other => return Err(format!("{other}= is not one of the limits")),
            }
        }
        match (burst, &mut limits.rate) {
            (Some(b), Some(rate)) => rate.burst = b,
            (Some(_), None) => return Err("burst= needs a rate= beside it".to_owned()),
            (None, _) => {}
        }
        Ok(limits)
    }
}

/// `n`, the count in field `name=value`, as a number from 1 to [`MOST`], or
/// an error naming the field.
fn count(name: &str, value: &str, n: &str) -> Result<u32, String> {
    n.parse()
        .ok()
        .filter(|n| (1..=MOST).contains(n))
        .ok_or_else(|| format!("{name}={value}: takes a whole number from 1 to {MOST}"))
}

/// One key's limits and what it holds of them now, shared by every request
/// the key makes.
#[derive(Debug)]
pub(crate) struct Meter {
    limits: Limits,
    held: Mutex<Held>,
}

#[derive(Debug, Default)]
struct Held {
    /// The key's requests being answered now.
    inflight: u32,
    /// When the bucket is next full again, counted one interval per request
    /// taken; `None` (or a time passed) is a full bucket.
    full_at: Option<Instant>,
}

/// Why a request was refused, with how long the client should wait.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// The key already has its `inflight=N` requests answered.
    Inflight(u32),
    /// The key's bucket is empty; one request's room returns after the wait.
    Rate { limits: String, wait: Duration },
}

impl Refusal {
    /// The `Retry-After` seconds: the wait rounded up, at least 1 (a place
    /// frees when one of the key's requests ends).
    pub(crate) fn retry_after(&self) -> u64 {
        match self {
            Refusal::Inflight(_) => 1,
            Refusal::Rate { wait, .. } => {
                (wait.as_secs() + u64::from(wait.subsec_nanos() > 0)).max(1)
            }
        }
    }

    /// The 429's message, naming the limit it met.
    pub(crate) fn message(&self) -> String {
        match self {
            Refusal::Inflight(n) => format!(
                "this API key is at its in-flight limit (inflight={n}); retry when one of its \
                 requests ends"
            ),
            Refusal::Rate { limits, .. } => format!(
                "this API key's request rate ({limits}) is spent; retry after {} s",
                self.retry_after()
            ),
        }
    }
}

/// One share of an admitted request's place, held until every share drops.
pub(crate) type Lease = Arc<Place>;

/// One counted place shared by the HTTP writer and any background worker.
#[derive(Debug)]
pub(crate) struct Place(Arc<Meter>);

impl Drop for Place {
    fn drop(&mut self) {
        let mut held = relock(&self.0.held);
        held.inflight -= 1;
    }
}

impl Meter {
    pub(crate) fn new(limits: Limits) -> Meter {
        Meter {
            limits,
            held: Mutex::default(),
        }
    }

    pub(crate) fn limits(&self) -> Limits {
        self.limits
    }

    /// A request of this key arriving at `now`: its lease when it `takes_place`
    /// among the key's in-flight requests (`None` when it does not), or the
    /// refusal of the first limit it meets. The in-flight limit is read
    /// first, so a request it refuses takes nothing from the bucket; a request
    /// that takes no place meets only the bucket.
    pub(crate) fn admit(
        self: &Arc<Self>,
        now: Instant,
        takes_place: bool,
    ) -> Result<Option<Lease>, Refusal> {
        let mut held = relock(&self.held);
        if let Some(n) = self.limits.inflight
            && takes_place
            && held.inflight >= n
        {
            return Err(Refusal::Inflight(n));
        }
        if let Some(r) = self.limits.rate {
            // The bucket as a time: it is full at `full_at`, each request
            // moves that an interval later, and a request is refused while it
            // lies past `burst - 1` intervals from now.
            let full_at = held.full_at.filter(|&t| t > now).unwrap_or(now);
            let room = r.interval * (r.burst - 1);
            let wait = full_at.saturating_duration_since(now).saturating_sub(room);
            if !wait.is_zero() {
                let unit = if r.per_minute { "m" } else { "s" };
                return Err(Refusal::Rate {
                    limits: format!("rate={}/{unit} burst={}", r.count, r.burst),
                    wait,
                });
            }
            held.full_at = Some(full_at + r.interval);
        }
        if !takes_place {
            return Ok(None);
        }
        held.inflight += 1;
        Ok(Some(Arc::new(Place(Arc::clone(self)))))
    }
}

#[cfg(test)]
mod tests {
    use super::{Limits, Meter, Refusal};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    fn meter(fields: &[(&str, &str)]) -> Arc<Meter> {
        Arc::new(Meter::new(Limits::parse(fields.iter().copied()).unwrap()))
    }

    /// The fields' spellings and their refusals, each naming the field and
    /// never anything else of the line.
    #[test]
    fn the_limit_fields_take_counts_and_a_unit_and_refuse_the_rest() {
        assert_eq!(Limits::parse([]), Ok(Limits::default()));
        let l = Limits::parse([("rate", "30/m"), ("burst", "5"), ("inflight", "2")]).unwrap();
        assert_eq!(l.inflight, Some(2));
        let r = l.rate.unwrap();
        assert_eq!((r.interval, r.burst), (Duration::from_secs(2), 5));
        let r = Limits::parse([("rate", "4/s")]).unwrap().rate.unwrap();
        assert_eq!((r.interval, r.burst), (Duration::from_millis(250), 4));
        for (fields, error) in [
            (
                vec![("inflight", "0")],
                "inflight=0: takes a whole number from 1 to 1000000",
            ),
            (
                vec![("inflight", "-1")],
                "inflight=-1: takes a whole number from 1 to 1000000",
            ),
            (
                vec![("burst", "1000001"), ("rate", "1/s")],
                "burst=1000001: takes a whole number",
            ),
            (vec![("rate", "10")], "rate=10: takes N/s or N/m"),
            (vec![("rate", "10/h")], "rate=10/h: the unit is s or m"),
            (vec![("rate", "0/s")], "rate=0/s: takes a whole number"),
            (vec![("burst", "3")], "burst= needs a rate= beside it"),
            (vec![("speed", "3")], "speed= is not one of the limits"),
        ] {
            let got = Limits::parse(fields.iter().copied());
            assert!(
                got.as_ref().is_err_and(|e| e.starts_with(error)),
                "{fields:?}: {got:?}"
            );
        }
    }

    /// `inflight=2`: a third request while two are held is a 429 with
    /// `Retry-After: 1`, and a lease's last share dropped gives its place back.
    #[test]
    fn the_inflight_limit_counts_the_leases_held() {
        let m = meter(&[("inflight", "2")]);
        let now = Instant::now();
        let a = m.admit(now, true).unwrap();
        let _b = m.admit(now, true).unwrap();
        let refused = m.admit(now, true).unwrap_err();
        assert_eq!(refused, Refusal::Inflight(2));
        assert_eq!(refused.retry_after(), 1);
        assert_eq!(
            refused.message(),
            "this API key is at its in-flight limit (inflight=2); retry when one of its requests ends"
        );
        let shared = a.clone();
        drop(a);
        assert_eq!(m.admit(now, true).unwrap_err(), Refusal::Inflight(2));
        drop(shared);
        assert!(m.admit(now, true).is_ok());
    }

    /// `rate=60/m burst=2`: two at once, the third waits the one second a
    /// request's room takes to come back, and a quiet spell refills the
    /// bucket to its burst and no further.
    #[test]
    fn the_rate_limit_is_a_bucket_of_its_burst() {
        let m = meter(&[("rate", "60/m"), ("burst", "2")]);
        let t0 = Instant::now();
        drop(m.admit(t0, true).unwrap());
        drop(m.admit(t0, true).unwrap());
        let refused = m.admit(t0 + Duration::from_millis(300), true).unwrap_err();
        assert_eq!(
            refused,
            Refusal::Rate {
                limits: "rate=60/m burst=2".to_owned(),
                wait: Duration::from_millis(700)
            }
        );
        assert_eq!(refused.retry_after(), 1);
        assert_eq!(
            refused.message(),
            "this API key's request rate (rate=60/m burst=2) is spent; retry after 1 s"
        );
        drop(m.admit(t0 + Duration::from_secs(1), true).unwrap());
        let later = t0 + Duration::from_secs(60);
        drop(m.admit(later, true).unwrap());
        drop(m.admit(later, true).unwrap());
        assert!(m.admit(later, true).is_err(), "the burst caps the refill");

        // A slow rate rounds its wait up to whole seconds.
        let slow = meter(&[("rate", "1/m")]);
        drop(slow.admit(t0, true).unwrap());
        assert_eq!(
            slow.admit(t0 + Duration::from_millis(500), true)
                .unwrap_err()
                .retry_after(),
            60
        );
    }

    /// A request the in-flight limit refuses takes nothing from the bucket,
    /// and a request that takes no place passes the in-flight limit and
    /// meets only the bucket.
    #[test]
    fn an_inflight_refusal_spends_no_rate() {
        let m = meter(&[("inflight", "1"), ("rate", "3/m")]);
        let now = Instant::now();
        let held = m.admit(now, true).unwrap();
        assert_eq!(m.admit(now, true).unwrap_err(), Refusal::Inflight(1));
        assert!(m.admit(now, false).unwrap().is_none(), "no place taken");
        drop(held);
        assert!(
            m.admit(now, true).is_ok(),
            "the refusal left the third token"
        );
        assert!(matches!(m.admit(now, false), Err(Refusal::Rate { .. })));
    }
}
