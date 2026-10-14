#!/usr/bin/env bash
# shellcheck shell=bash
# The exit rule of a grep whose empty result is an answer ("nothing found"), for the check scripts under tools/.
# Sourced, never executed; one owner, so every check reads grep's status the same way.
#
# grep exits 1 when no line matches and 2 when it cannot read a file or the pattern is bad, and in the second case it
# still prints the lines of the files it did read. A check that turns either status into success reads an unreadable
# file as one with no hit. Here 1 is an answer and returns 0, so in a pipeline under pipefail a nonzero status is an
# error of some stage, never an empty result.
#
#   grep_answer <tool> <grep arguments…>   grep's lines and 0 on a match or none, or a line on stderr and grep's status
grep_answer() {
  local tool=$1 rc=0
  shift
  grep "$@" || rc=$?
  [ "$rc" -le 1 ] || { echo "$tool: grep exit $rc (its message above), no verdict" >&2; return "$rc"; }
}
