//! The run order and the verdict of a gate binary whose gates are modules of
//! it. The binary's arguments name the gates to run, each one of its own;
//! none runs them all, in the binary's order. A gate that fails does not stop
//! the next one: each prints its own result, and the process ends red when
//! any of them failed.

use std::process::ExitCode;
use std::time::Instant;

use bloomery_gpu_gates::{GateError, exit_with};

/// Runs the gates of the binary `bin`. `all` holds their names in run order,
/// `args` the names the caller asked for (none: every gate), and `run` runs
/// the one gate it is given. A name that is not in `all` is refused by name
/// before any gate runs. A failed gate's error is printed under
/// `{prefix}{name}`, the name the gate prints its own lines under; the status
/// is the first failure's.
pub fn run_set(
    bin: &str,
    prefix: &str,
    all: &[&str],
    args: impl Iterator<Item = String>,
    mut run: impl FnMut(&str) -> Result<(), GateError>,
) -> ExitCode {
    let asked: Vec<String> = args.collect();
    if let Some(bad) = asked.iter().find(|a| !all.contains(&a.as_str())) {
        return exit_with(
            bin,
            Err(format!("no gate named {bad:?}; the gates are {all:?}").into()),
        );
    }
    let picked: Vec<&str> = if asked.is_empty() {
        all.to_vec()
    } else {
        asked.iter().map(String::as_str).collect()
    };
    let mut status = None;
    let mut failed = Vec::new();
    for &name in &picked {
        println!("{bin}: gate {name}");
        let start = Instant::now();
        let result = run(name);
        let verdict = if result.is_ok() { "ok" } else { "FAILED" };
        println!(
            "{prefix}{name}: {verdict} in {:.1} s",
            start.elapsed().as_secs_f64()
        );
        if let Err(e) = result {
            failed.push(name);
            let code = exit_with(&format!("{prefix}{name}"), Err(e));
            status = status.or(Some(code));
        }
    }
    println!(
        "{bin}: {} of {} gates passed",
        picked.len() - failed.len(),
        picked.len()
    );
    if !failed.is_empty() {
        eprintln!("{bin}: failed: {failed:?}");
    }
    status.unwrap_or(ExitCode::SUCCESS)
}
