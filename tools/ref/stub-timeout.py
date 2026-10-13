#!/usr/bin/env python3
"""A stand-in timeout for the self-tests of tools/gpu-gate.sh and tools/ref/load-groups.sh where the host has none (the
Mac), copied in as `timeout`. It takes coreutils' `timeout [--kill-after=K] S COMMAND...`: at S seconds COMMAND is killed
and the exit is 124; a TERM sent here goes on to COMMAND, whose status is the exit (128 + the signal). K is ignored."""
import signal
import subprocess
import sys

args = [a for a in sys.argv[1:] if not a.startswith('--kill-after=')]
proc = subprocess.Popen(args[1:])
signal.signal(signal.SIGTERM, lambda *_: proc.terminate())
try:
    rc = proc.wait(timeout=float(args[0]))
except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait()
    sys.exit(124)
sys.exit(128 - rc if rc < 0 else rc)
