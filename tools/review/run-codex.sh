#!/usr/bin/env bash
# One headless Codex CLI run of the review bot: run-claude.sh's counterpart, with its arguments, prompt, trace files
# and exit code rule. tools/review/run.py picks the engine and stops a run that overruns its minutes; a maintainer can
# call either script by hand.
#
#   tools/review/run-codex.sh reviewer|validator|learner BASE_TREE PR_TREE CTX OUT TRACE
#
# The login is CODEX_HOME/auth.json (what `codex login` writes), which the workflow puts outside the trees and OUT.
# Codex reads files by running shell commands, and the PR's text can steer which commands the model runs, so the
# sandbox is what keeps the run inside. A permission profile lets a command read the PR tree, the base tree, CTX, the
# Codex install and the system paths of `:minimal` (on Linux /bin, /sbin, /usr, /etc, /lib, /lib64; on macOS also /tmp
# and /var/tmp), write OUT only, and use no network. On Linux a command can also write the empty / and /dev that
# bubblewrap mounts for it, which no process outside sees and which go when the command ends. The legacy
# `--sandbox workspace-write` is not used: it lets a command read every file the runner user can, auth.json included.
# The working directory is OUT; no AGENTS.md is loaded (project_doc_max_bytes=0), no parent directory is a project
# root, and --ignore-user-config and --ignore-rules skip CODEX_HOME's config and rules, so nothing in the PR tree is
# configuration. Commands get the core environment only. Hooks, plugins, memories, apps, image generation, web search
# and sub-agents are off.
#
# Before the run, `codex sandbox` runs seven commands under the same profile: a read in BASE_TREE and a write to OUT
# must work, and a read of a file beside auth.json, a write into PR_TREE, /tmp or /var/tmp and a connection to
# 1.1.1.1:443 must not.
# When one of them goes the wrong way the run does not start: the script prints why and exits 69.
# The JSONL event stream goes to TRACE/<role>.ndjson and tools/review/usage.py's accounting to TRACE/<role>.usage.json.
# The exit code is codex's, and a stream without a completed turn is exit 1.
set -euo pipefail

role=${1:?role}; base=${2:?base tree}; pr=${3:?pr tree}; ctx=${4:?ctx}; out=${5:?out}; trace=${6:?trace}
case "$role" in
  reviewer|validator|learner) ;;
  *) echo "run-codex.sh: role is reviewer, validator or learner, not '$role'" >&2; exit 64 ;;
esac
for d in "$base" "$pr" "$ctx" "$out" "$trace"; do
  case "$d" in /*) ;; *) echo "run-codex.sh: '$d' is not an absolute path" >&2; exit 64 ;; esac
  case "$d" in
    *'"'*|*\\*|*'}'*|*=*|*,*) echo "run-codex.sh: '$d' has a quote, backslash, brace, = or comma" >&2; exit 64 ;;
  esac
  [ -d "$d" ] || { echo "run-codex.sh: no directory $d" >&2; exit 64; }
done
instructions="$base/.github/review/$role.md"
[ -f "$instructions" ] || { echo "run-codex.sh: no $instructions" >&2; exit 66; }

prompt=$(cat "$instructions"; printf '\n## Paths for this run\n\n- PR_TREE: %s\n- BASE_TREE: %s\n- CTX: %s\n- OUT: %s\n' \
  "$pr" "$base" "$ctx" "$out")

# The sandbox runs Codex's own binary for file edits and instruction loading, so its path must be readable: the
# command on PATH and the directory above its package (an npm install keeps the platform binary beside it).
bin=$(command -v codex)
pkg=$(dirname "$(dirname "$(readlink -f "$bin")")")
fs="\":minimal\"=\"read\",\"$(dirname "$bin")\"=\"read\",\"$(dirname "$pkg")\"=\"read\""
fs="$fs,\"$pr\"=\"read\",\"$base\"=\"read\",\"$ctx\"=\"read\",\"$out\"=\"write\""
# On macOS :minimal makes /tmp and /var/tmp writable; an explicit entry keeps them read-only.
[ "$(uname -s)" != Darwin ] || fs="$fs,\"/tmp\"=\"read\",\"/var/tmp\"=\"read\""
profile=(
  -c default_permissions='"review"'
  -c "permissions.review.filesystem={$fs}"
  -c permissions.review.network.enabled=false
  -c project_doc_max_bytes=0
  -c project_root_markers='[]'
  -c shell_environment_policy.inherit='"core"'
)

home=${CODEX_HOME:-$HOME/.codex}
[ -d "$home" ] || { echo "run-codex.sh: no $home (codex login writes it)" >&2; exit 66; }
probe="$home/sandbox-check.$$"
trap 'rm -f "$probe"' EXIT
seal=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
printf '%s\n' "$seal" > "$probe"
sbx() { (cd "$out" && codex "${profile[@]}" sandbox -- "$@" 2> /dev/null < /dev/null) || true; }
why=""
[ "$(sbx /bin/cat "$instructions")" = "$(cat "$instructions")" ] || why="a read in BASE_TREE failed"
sbx /usr/bin/touch "$out/.sandbox-check"
[ -f "$out/.sandbox-check" ] || why="${why:+$why; }a write to OUT failed"
rm -f "$out/.sandbox-check"
case "$(sbx /bin/cat "$probe")" in *"$seal"*) why="${why:+$why; }a file beside auth.json was readable" ;; esac
sbx /usr/bin/touch "$pr/.sandbox-check"
if [ -e "$pr/.sandbox-check" ]; then why="${why:+$why; }PR_TREE was writable"; rm -f "$pr/.sandbox-check"; fi
for d in /tmp /var/tmp; do
  sbx /usr/bin/touch "$d/.sandbox-check.$$"
  if [ -e "$d/.sandbox-check.$$" ]; then why="${why:+$why; }$d was writable"; rm -f "$d/.sandbox-check.$$"; fi
done
[ -z "$(sbx /bin/bash -c 'exec 3<>/dev/tcp/1.1.1.1/443 && echo open')" ] || why="${why:+$why; }the network was open"
if [ -n "$why" ]; then
  echo "run-codex.sh: sandbox check failed: $why" >&2
  exit 69
fi
echo "run-codex.sh: sandbox check passed" >&2

transcript="$trace/$role.ndjson"
rc=0
(
  cd "$out"
  codex exec "$prompt" < /dev/null \
    --json \
    --model "${REVIEW_CODEX_MODEL:-gpt-6.1-sol}" \
    -c model_reasoning_effort='"xhigh"' \
    ${REVIEW_CODEX_TIER:+-c "service_tier=\"$REVIEW_CODEX_TIER\""} \
    --cd "$out" \
    --skip-git-repo-check --ephemeral --ignore-user-config --ignore-rules \
    -c approval_policy='"never"' \
    "${profile[@]}" \
    -c web_search='"disabled"' \
    -c features.multi_agent=false \
    -c features.apps=false \
    -c features.hooks=false \
    -c features.plugins=false \
    -c features.memories=false \
    -c features.image_generation=false \
    -c check_for_update_on_startup=false \
    -c analytics.enabled=false
) > "$transcript" || rc=$?

prc=0
python3 "$base/tools/review/usage.py" "$role" "$transcript" "$trace/$role.usage.json" codex || prc=$?
[ "$rc" -ne 0 ] || rc=$prc
exit "$rc"
