#!/usr/bin/env bash
# Fixtures shared by the self-tests of tools whose md5sum or wc reads must fail by name (the reference-set
# manifest writers, the PTX scanners). Sourced, never executed.

# read_stubs <dir>: md5sum and wc stand-ins in <dir>, for the front of PATH. Each reads its stdin and
# fails with no output when the text holds the word `fail-md5` or `fail-wc`; otherwise md5sum prints one
# fixed digest and wc prints 7, on any host (a Mac has no md5sum, and its wc pads the count).
read_stubs() {
  printf '#!/bin/sh\n! grep -qw fail-md5 || exit 1\necho "0123456789abcdef0123456789abcdef  -"\n' > "$1/md5sum"
  printf '#!/bin/sh\n! grep -qw fail-wc || exit 1\necho 7\n' > "$1/wc"
  chmod +x "$1/md5sum" "$1/wc"
}

# check <name> <got> <want>: prints ok or FAIL and counts failures in the caller's `bad`.
check() { if [ "$2" = "$3" ]; then echo "ok $1"; else echo "FAIL $1: got [$2], want [$3]"; bad=$((bad + 1)); fi; }
