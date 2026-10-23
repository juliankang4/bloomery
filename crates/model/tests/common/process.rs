pub(crate) fn in_process(test: &str, env: (&str, &str)) -> bool {
    const FRESH: &str = "a-fresh-process-for-model-tests";
    if std::env::args().any(|a| a == FRESH) {
        return true;
    }
    let (_, test) = test.split_once("::").expect("a crate path");
    let out = std::process::Command::new(std::env::current_exe().expect("the test binary"))
        .args(["--exact", test, FRESH])
        .env(env.0, env.1)
        .output()
        .expect("the fresh process");
    assert!(
        out.status.success() && String::from_utf8_lossy(&out.stdout).contains("1 passed"),
        "the fresh process: {}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    false
}
