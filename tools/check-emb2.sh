#!/usr/bin/env bash
# EmbeddingGemma writer contracts, with local shell fixtures and no model or card.
# --dumper PATH also checks the linked dumper's thread option before it opens a model.
# No argument: the shell contracts; a missing Linux utility is a named skip.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 - "$@" <<'PY'
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time
import unittest

parser = argparse.ArgumentParser()
parser.add_argument("--dumper", type=Path)
args = parser.parse_args()
ROOT = Path.cwd()
DUMPER = args.dumper.resolve() if args.dumper else None


def write(path, text, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(0o755)


class Writer(unittest.TestCase):
    def setUp(self):
        for name in ("flock", "timeout", "sha256sum", "readlink"):
            if not shutil.which(name):
                self.skipTest(f"{name} is required for the Linux writer fixtures")
        self.root = Path(tempfile.mkdtemp(prefix="bloomery-emb2-")).resolve()
        self.addCleanup(shutil.rmtree, self.root)
        for name in ("tools/ref/emb2/emb2.sh", "tools/gate-bound.sh",
                     "tools/ref/emb2/dump_emb2.cpp", "tools/ref/clefvis/set_writer.h"):
            dest = self.root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        self.data = self.root / "data"
        self.data.mkdir()
        self.final = self.data / "ref_fixture_tower.cpu"
        self.oracle = self.root / "oracle.tree"
        self.oracle.mkdir()
        self.pin = "a" * 40
        model, projector = self.root / "text.gguf", self.root / "projector.gguf"
        model.write_bytes(b"text fixture")
        projector.write_bytes(b"projector fixture")
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        variables = {
            "DATA": self.data, "MODEL": model, "MMPROJ": projector,
            "MODEL_SHA256": digest(model), "MMPROJ_SHA256": digest(projector),
            "LCPP_PIN": self.pin, "MODEL_NAME": "gemma-embedding2",
            "ORACLE": self.oracle, "BIN": self.root / "dump", "FAMILY": "emb2",
            "DUMPER": "tools/ref/emb2/dump_emb2.cpp", "SETPFX": "ref_fixture", "THREADS": "1",
            "EMBED_VOCAB": "fixture", "EMBED_CASES": "cases-fixture.txt", "EMBED_CONTEXT": "8192",
        }
        write(self.root / "tools/ref/ref-paths.sh", "\n".join(
            f"{key}={shlex.quote(str(value))}" for key, value in variables.items()) + "\n")
        write(self.root / "tools/ref/clefvis/profile.sh", "IMAGES=()\nEMBED_STEPS=(text tower server text.cpu tower.cpu)\n")
        write(self.oracle / "build/bin/llama-server", f"#!/bin/bash\necho 'version: fixture (commit {self.pin})'\n", True)
        self.flock = shutil.which("flock")
        write(self.root / "bin/flock", "#!/bin/bash\n"
              'if [ "$1" = -w ]; then touch "$FIXTURE_ATTEMPT/$FIXTURE_TAG"; fi\n'
              + "exec " + shlex.quote(self.flock) + ' "$@"\n', True)
        (self.root / "attempts").mkdir()
        write(self.root / "bin/git", """#!/bin/bash
case " $* " in
  *' rev-parse HEAD '*) printf '%s\\n' "$FIXTURE_PIN" ;;
  *' status '*) exit "${FIXTURE_STATUS_RC:-0}" ;;
  *) exit 91 ;;
esac
""", True)
        write(self.root / "dump", """#!/bin/bash
set -eu
out=
while [ "$#" -gt 0 ]; do
  if [ "$1" = --out ]; then out=$2; shift; fi
  shift
done
printf '%s:start\\n' "$FIXTURE_TAG" >> "$FIXTURE_EVENTS"
if [ -n "${FIXTURE_RELEASE:-}" ]; then
  for _ in {1..200}; do
    [ ! -e "$FIXTURE_RELEASE" ] || break
    sleep 0.02
  done
  [ -e "$FIXTURE_RELEASE" ] || exit 92
fi
[ "${FIXTURE_FAIL:-0}" = 0 ] || exit 42
mkdir -p "$out"
printf '%s\\n' "$FIXTURE_TAG" > "$out/MANIFEST.tsv"
printf '%s:end\\n' "$FIXTURE_TAG" >> "$FIXTURE_EVENTS"
""", True)
        record = [
            "source_sha256 " + digest(self.root / "tools/ref/emb2/dump_emb2.cpp"),
            "set_writer_sha256 " + digest(self.root / "tools/ref/clefvis/set_writer.h"),
            "lcpp_commit " + self.pin[:9], "lcpp " + str(self.oracle.resolve()),
        ]
        write(self.root / "dump.build", "\n".join(record) + "\n")
        self.env = dict(os.environ, PATH=str(self.root / "bin") + os.pathsep + os.environ["PATH"],
                        EMB2_BOUND="8", FIXTURE_ATTEMPT=str(self.root / "attempts"), FIXTURE_PIN=self.pin, FIXTURE_EVENTS=str(self.root / "events"))
        self.jobs = []
        self.addCleanup(self.stop_jobs)

    def stop_jobs(self):
        for process in self.jobs:
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=12)

    def start(self, tag, **extra):
        env = dict(self.env, FIXTURE_TAG=tag, **extra)
        process = subprocess.Popen(["bash", "tools/ref/emb2/emb2.sh", "tower.cpu"],
                                   cwd=self.root, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        self.jobs.append(process)
        return process

    def finish(self, process, rc=0):
        output, _ = process.communicate(timeout=12)
        self.assertEqual(process.returncode, rc, output)
        return output

    def wait_attempt(self, tag):
        for _ in range(200):
            if (self.root / "attempts" / tag).exists():
                return
            time.sleep(0.02)
        self.fail(f"{tag} did not attempt the set lock")

    def wait_started(self):
        for _ in range(200):
            if (self.root / "events").exists():
                return
            time.sleep(0.02)
        self.fail("the first writer did not reach generation")

    def test_text_only_profile_refuses_tower_and_joined_steps_before_the_runner(self):
        env = os.environ.copy()
        env.pop("BLOOMERY_MODEL", None)
        env["BLOOMERY_REF_MODEL_PROFILE"] = "gemma-embedding"
        for step in ("tower", "text server"):
            with self.subTest(step=step):
                result = subprocess.run(["bash", str(ROOT / "tools/ref/emb2/emb2.sh"), step],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 64, result.stderr)
                self.assertIn(f"unknown step '{step}'", result.stderr)

    def test_lock_precedes_recovery_and_staging(self):
        old = Path(str(self.final) + ".old")
        stage = Path(str(self.final) + ".staging")
        write(old / "MANIFEST.tsv", "old\n")
        write(stage / "sentinel", "staging\n")
        with Path(str(self.final) + ".lock").open("w") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            process = self.start("new")
            self.wait_attempt("new")
            time.sleep(0.1)
            self.assertIsNone(process.poll())
            self.assertFalse(self.final.exists())
            self.assertEqual((old / "MANIFEST.tsv").read_text(), "old\n")
            self.assertEqual((stage / "sentinel").read_text(), "staging\n")
            self.assertFalse((self.root / "events").exists())
            fcntl.flock(held, fcntl.LOCK_UN)
        self.finish(process)
        self.assertEqual((self.final / "MANIFEST.tsv").read_text(), "new\n")
        self.assertFalse(old.exists())

    def test_contended_lock_names_the_set_and_returns_75(self):
        write(self.final / "MANIFEST.tsv", "old\n")
        with Path(str(self.final) + ".lock").open("w") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            output = self.finish(self.start("blocked", EMB2_BOUND="1"), 75)
        self.assertIn("set lock for " + str(self.final), output)
        self.assertEqual((self.final / "MANIFEST.tsv").read_text(), "old\n")
        self.assertFalse((self.root / "events").exists())

    def test_two_writers_generate_and_publish_in_order(self):
        write(self.final / "MANIFEST.tsv", "old\n")
        release = self.root / "release"
        first = self.start("first", FIXTURE_RELEASE=str(release))
        self.wait_started()
        second = self.start("second")
        self.wait_attempt("second")
        time.sleep(0.1)
        self.assertEqual((self.root / "events").read_text(), "first:start\n")
        self.assertEqual((self.final / "MANIFEST.tsv").read_text(), "old\n")
        release.touch()
        self.finish(first)
        self.finish(second)
        self.assertEqual((self.root / "events").read_text(),
                         "first:start\nfirst:end\nsecond:start\nsecond:end\n")
        self.assertEqual((self.final / "MANIFEST.tsv").read_text(), "second\n")
        self.assertFalse(Path(str(self.final) + ".staging").exists())
        self.assertFalse(Path(str(self.final) + ".old").exists())

    def test_failed_writer_keeps_the_set_and_releases_the_lock(self):
        write(self.final / "MANIFEST.tsv", "old\n")
        self.finish(self.start("failed", FIXTURE_FAIL="1"), 42)
        self.assertEqual((self.final / "MANIFEST.tsv").read_text(), "old\n")
        self.finish(self.start("next"))
        self.assertEqual((self.final / "MANIFEST.tsv").read_text(), "next\n")

    def test_failed_status_check_stops_before_generation(self):
        output = self.finish(self.start("failed-status", FIXTURE_STATUS_RC="128"), 2)
        self.assertIn("cannot inspect tracked changes in " + str(self.oracle), output)
        self.assertFalse((self.root / "events").exists())

    def test_build_record_from_another_tree_is_refused(self):
        record = self.root / "dump.build"
        record.write_text(record.read_text().replace("lcpp " + str(self.oracle),
                                                     "lcpp " + str(self.oracle).replace("oracle.tree", "oracleXtree")))
        output = self.finish(self.start("wrong-tree"), 2)
        self.assertIn("no 'lcpp " + str(self.oracle), output)
        self.assertFalse((self.root / "events").exists())


class Builders(unittest.TestCase):
    def test_unreadable_index_stops_before_compilation(self):
        scripts = (
            ("tools/ref/build-hidden.sh", "qwen35"),
            ("tools/ref/build-nemotron-ref.sh", "nemotron_h"),
            ("tools/ref/clefvis/build-clefvis.sh", "qwen35"),
            ("tools/ref/qvis/build-lcpp-qvis.sh", "qwen35"),
        )
        for script, profile in scripts:
            with self.subTest(script=script), tempfile.TemporaryDirectory(prefix="oracle-status-") as tmp:
                root = Path(tmp).resolve()
                target = root / script
                target.parent.mkdir(parents=True)
                shutil.copyfile(ROOT / script, target)
                oracle = root / "oracle"
                subprocess.run(["git", "init", "-q", str(oracle)], check=True)
                subprocess.run(["git", "-C", str(oracle), "-c", "user.name=fixture",
                                "-c", "user.email=fixture@example.invalid", "commit",
                                "--allow-empty", "-qm", "fixture"], check=True)
                pin = subprocess.check_output(["git", "-C", str(oracle), "rev-parse", "HEAD"],
                                              text=True).strip()
                (oracle / ".git/index").write_bytes(b"invalid index")
                write(root / "tools/ref/ref-paths.sh",
                      f"MODEL_NAME={profile}\nLCPP={shlex.quote(str(oracle))}\n"
                      f"BLOOMERY_DATA={shlex.quote(str(root / 'data'))}\n")
                write(root / "tools/ref/clefvis/profile.sh",
                      'ORACLE=$LCPP\nDATA=$BLOOMERY_DATA\nDUMPER=fixture.cpp\n')
                write(root / "tools/ref/lease.sh", 'lease_free() { return 0; }\nnow() { printf fixture; }\n')
                marker = root / "compiled"
                for compiler in ("g++", "cmake"):
                    write(root / "cuda/bin" / compiler,
                          '#!/bin/bash\n: > "$FIXTURE_COMPILED"\nexit 88\n', True)
                # Exercise source integrity without qvis' root-only user switch.
                write(root / "cuda/bin/id", "#!/bin/bash\necho 501\n", True)
                env = dict(os.environ, PATH=str(root / "cuda/bin") + os.pathsep + os.environ["PATH"],
                           LCPP_BUILD_TREE=str(oracle), LCPP_BUILD_PIN=pin,
                           LCPP_BUILD_DIR=str(root / "run"), LCPP_BUILD_CUDA=str(root / "cuda"),
                           FIXTURE_COMPILED=str(marker))
                result = subprocess.run(["bash", str(target)], cwd=root, env=env,
                                        capture_output=True, text=True, timeout=10)
                output = result.stdout + result.stderr
                if script == "tools/ref/qvis/build-lcpp-qvis.sh":
                    output += (root / "run/log").read_text()
                    self.assertEqual((root / "run/rc").read_text().strip(), "2", output)
                self.assertEqual(result.returncode, 2, output)
                self.assertIn("cannot inspect tracked changes", output)
                self.assertFalse(marker.exists(), output)


class Options(unittest.TestCase):
    def test_nonroot_build_recipe_names_the_failure_before_detaching(self):
        recipe = (ROOT / "justfile").read_text().split("\nbuild-emb2-lcpp:\n", 1)[1].split("\n\n", 1)[0]
        words = shlex.split(recipe.strip())
        command = words[words.index("./tools/box.sh") + 1]
        with tempfile.TemporaryDirectory(prefix="bloomery-emb2-build-") as tmp:
            tmp = Path(tmp)
            write(tmp / "id", "#!/bin/bash\necho 501\n", True)
            marker = tmp / "detached"
            write(tmp / "setsid", "#!/bin/bash\ntouch \"$FIXTURE_DETACHED\"\n", True)
            env = dict(os.environ, PATH=str(tmp) + os.pathsep + os.environ["PATH"], FIXTURE_DETACHED=str(marker))
            result = subprocess.run(["bash", "-c", command], env=env, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("build-emb2-lcpp: this recipe requires root", result.stderr)
            self.assertFalse(marker.exists())

    @unittest.skipUnless(DUMPER, "--dumper PATH is needed for the linked C++ option checks")
    def test_thread_counts_are_checked_before_model_loading(self):
        for value in ("", "garbage", "16junk", "-1", "2147483648", "999999999999999999999999999999999"):
            with self.subTest(value=value):
                result = subprocess.run([str(DUMPER), "text", "-t", value], capture_output=True, text=True, timeout=5)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("-t requires an integer from 0 to INT_MAX", result.stderr)
        for value in ("0", "1", "16", "2147483647"):
            with self.subTest(value=value):
                result = subprocess.run([str(DUMPER), "text", "-t", value], capture_output=True, text=True, timeout=5)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("usage: dump_emb2", result.stderr)
                self.assertNotIn("-t requires", result.stderr)


class Ids(unittest.TestCase):
    """set_writer.h's parse_ids, built alone: an id file is digits only, one id from 0 to INT32_MAX a line."""

    @classmethod
    def setUpClass(cls):
        cxx = shutil.which("c++") or shutil.which("g++")
        if not cxx:
            raise unittest.SkipTest("a C++ compiler is required for the id reader checks")
        cls.tmp = Path(tempfile.mkdtemp(prefix="bloomery-emb2-ids-"))
        # gguf_arch needs gguf.h's four calls; this build never calls it, so they are stubs and llama.cpp is not needed.
        gguf = ("#include <cstdint>\nstruct gguf_context;\nstruct gguf_init_params { bool no_alloc; void * ctx; };\n"
                "static gguf_context * gguf_init_from_file(const char *, gguf_init_params) { return nullptr; }\n"
                "static int64_t gguf_find_key(const gguf_context *, const char *) { return -1; }\n"
                'static const char * gguf_get_val_str(const gguf_context *, int64_t) { return ""; }\n'
                "static void gguf_free(gguf_context *) {}\n")
        write(cls.tmp / "ids.cpp", gguf + '#define DUMP_TOOL "ids"\n#include "' + str(ROOT / "tools/ref/clefvis/set_writer.h") + '"\n'
              "int main(int, char ** argv) {\n    std::vector<uint8_t> raw;\n    if (!read_file(argv[1], raw)) return 2;\n"
              '    for (int32_t id : parse_ids(argv[1], raw)) printf("%d\\n", id);\n}\n')
        cls.bin = cls.tmp / "ids"
        subprocess.run([cxx, "-std=c++17", "-o", str(cls.bin), str(cls.tmp / "ids.cpp")], check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def read(self, data):
        path = self.tmp / "case.ids"
        path.write_bytes(data)
        return path, subprocess.run([str(self.bin), str(path)], capture_output=True, timeout=5)

    def test_malformed_lines_name_the_file_and_line(self):
        for data in (b"7\n-1\n", b"7\n+1\n", b"7\n 1\n", b"7\n1 \n", b"7\n1\x002\n", b"7\n\n", b"7\n2147483648\n", b"7\n0x1\n"):
            with self.subTest(data=data):
                path, result = self.read(data)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(f"ids: {path}:2 is not a token id".encode(), result.stderr)
                self.assertEqual(result.stdout, b"")

    def test_digit_lines_are_read(self):
        _, result = self.read(b"0\n42\n2147483647\n")
        self.assertEqual((result.returncode, result.stdout), (0, b"0\n42\n2147483647\n"), result.stderr)


unittest.main(argv=["check-emb2"], verbosity=2)
PY
