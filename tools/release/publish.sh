#!/usr/bin/env bash
# Publishes a built release: `just release-build <version>` makes the tarball on the
# box; this script is everything after it, so the upload cannot repeat v0.2.0's
# mistake (gh named the assets from temp filenames and every download URL 404'd).
#
#   tools/release/publish.sh <version> [--dry-run] [--no-tap]
#
# What it owns, in order:
#   1. the tag: HEAD must carry tag v<version> (the release is the tag; the binaries'
#      --version names this commit);
#   2. the versions: the serve AND gpu-gates crates must both say <version> (--version
#      reads gpu-gates, /props reads serve — v0.2.0 shipped with 0.1.0 in --version);
#   3. the artifacts: on the box, where readelf and the ELF binary run — the release's
#      own .sha256, build.sh's checks on the unpacked binary, and its --version;
#      then the tarball and .sha256 come here for the upload and the tap's digest;
#      the GB10 archive (linux-aarch64-cuda-sm121a) the same way on the GB10 that built
#      it, named by BLOOMERY_GB10_BOX (unset: the release carries no GB10 archive, and
#      step 6 says so);
#   3b. the Apple Silicon artifacts, which tools/release/build-mac.sh wrote into this tree's
#      target/mac-release/dist/ from this commit: their .sha256, build-mac.sh's binary checks
#      on the unpacked binaries, a Developer ID signature on each, and the commit the
#      archive's README names;
#   4. the upload: `gh release upload --clobber` with exactly the names the build scripts
#      wrote (bloomery-<v>-linux-x86_64-cuda-sm86.tar.gz, bloomery-<v>-macos-arm64-metal.tar.gz,
#      bloomery-<v>-linux-aarch64-cuda-sm121a.tar.gz with a GB10 archive, and their .sha256) —
#      it refuses to upload a file under any other name — then curls every URL for a 200;
#   5. the Homebrew tap: bumps Formula/bloomery.rb's version and each platform's url and
#      sha256 in ~/repo/homebrew-tap (the GB10 one where the formula has its url), commits
#      and pushes;
#   6. prints what is left to the lead: the release notes (gh release edit --notes-file)
#      and the ghcr image — the v* tag push already triggered .github/workflows/container.yml.
#
# --dry-run does 1–3 and prints 4–6 without touching the release or the tap.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
VERSION=${1:?usage: tools/release/publish.sh <version> [--dry-run] [--no-tap]}
DRY=0; TAP=1
for a in "${@:2}"; do
    case "$a" in
        --dry-run) DRY=1 ;;
        --no-tap) TAP=0 ;;
        *) echo "publish.sh: unknown argument $a" >&2; exit 64 ;;
    esac
done
REPO=midagedev/bloomery
BOX=${BLOOMERY_BOX:-ws}
REMOTE=${BLOOMERY_REMOTE:-"~/repo/$(basename "$ROOT")"}
GB10_BOX=${BLOOMERY_GB10_BOX:-}
NAME="bloomery-$VERSION-linux-x86_64-cuda-sm86"
GB10_NAME="bloomery-$VERSION-linux-aarch64-cuda-sm121a"
MAC_NAME="bloomery-$VERSION-macos-arm64-metal"
TAG="v$VERSION"

fail() { echo "publish.sh: $*" >&2; exit 1; }

# 1. the tag
COMMIT=$(git rev-parse --short=12 HEAD)
[ -z "$(git status --porcelain --untracked-files=no)" ] || fail "the tree has uncommitted changes"
git tag -l --points-at HEAD | grep -qx "$TAG" || fail "HEAD does not carry $TAG (tag the release first)"

# 2. the versions: --version reads gpu-gates, /props reads serve
for c in serve gpu-gates; do
    v=$(sed -n 's/^version = "\(.*\)"/\1/p' "crates/$c/Cargo.toml" | head -1)
    [ "$v" = "$VERSION" ] || fail "crates/$c says version $v, not $VERSION — bump it, commit, re-tag, release-build again"
done

# 2b. the README names this release: its status heading, the tarball it unpacks and the release link
for want in "## Status ($VERSION)" "$NAME.tar.gz" "releases/tag/$TAG)"; do
    grep -qF -- "$want" README.md || fail "README.md does not name '$want' — update its status heading and install lines, commit, re-tag"
done

# 3. the Linux artifacts, each verified on the machine that built it (readelf/objdump and the ELF binary are Linux's)
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
# linux_archive <box> <name>: <name>'s .sha256, build.sh --check and --version on <box>; then the tarball and .sha256
# fetched into $TMP and hashed here. Prints the sha256.
linux_archive() {
    local box=$1 name=$2 sha got f
    local verify=(ssh "$box")
    # The box's command goes behind its guard; the GB10's runs over plain ssh, as the account that built it (box.sh's
    # guard reads /root's lease, which that account need not reach).
    [ "$box" != "$BOX" ] || verify=(bash tools/box.sh)
    BLOOMERY_BOX=$box "${verify[@]}" "set -e; cd $REMOTE/target/release-dist; \
sha256sum -c $name.tar.gz.sha256 > /dev/null; \
rm -rf /tmp/pub-$VERSION; mkdir /tmp/pub-$VERSION; \
tar -xzf $name.tar.gz -C /tmp/pub-$VERSION --strip-components=1; \
bash $REMOTE/tools/release/build.sh --check /tmp/pub-$VERSION/bin; \
V=\$(/tmp/pub-$VERSION/bin/bloomery-serve --version); \
[ \"\$V\" = \"bloomery-serve $VERSION (commit $COMMIT)\" ] || { echo \"the packed binary says: \$V\" >&2; exit 1; }" >&2 \
        || fail "$name failed verification on $box (just release-build $VERSION there first)"
    for f in "$name.tar.gz" "$name.tar.gz.sha256"; do
        scp -q "$box:$REMOTE/target/release-dist/$f" "$TMP/$f" || fail "cannot fetch $f from $box"
    done
    sha=$(grep -o '[0-9a-f]\{64\}' "$TMP/$name.tar.gz.sha256" | head -1)
    [ -n "$sha" ] || fail "$name's .sha256 holds no digest"
    got=$(shasum -a 256 "$TMP/$name.tar.gz" | awk '{print $1}')
    [ "$got" = "$sha" ] || fail "the fetched $name.tar.gz hashes $got, its .sha256 says $sha"
    echo "publish.sh: $name.tar.gz $(stat -f %z "$TMP/$name.tar.gz") bytes, sha256 $sha, checks and --version ok on $box" >&2
    echo "$sha"
}
SHA=$(linux_archive "$BOX" "$NAME") || exit 1
GB10_SHA=
if [ -n "$GB10_BOX" ]; then
    GB10_SHA=$(linux_archive "$GB10_BOX" "$GB10_NAME") || exit 1
fi

# 3b. the Apple Silicon artifacts, built here by tools/release/build-mac.sh
for f in "$MAC_NAME.tar.gz" "$MAC_NAME.tar.gz.sha256"; do
    [ -f "target/mac-release/dist/$f" ] || fail "no target/mac-release/dist/$f (MAC_SIGN_IDENTITY=… MAC_NOTARY_PROFILE=… tools/release/build-mac.sh $VERSION first)"
    cp "target/mac-release/dist/$f" "$TMP/$f"
done
MAC_SHA=$(grep -o '[0-9a-f]\{64\}' "$TMP/$MAC_NAME.tar.gz.sha256" | head -1)
[ "$(shasum -a 256 "$TMP/$MAC_NAME.tar.gz" | awk '{print $1}')" = "$MAC_SHA" ] || fail "$MAC_NAME.tar.gz does not hash to its .sha256"
mkdir "$TMP/mac" && tar -xzf "$TMP/$MAC_NAME.tar.gz" -C "$TMP/mac" --strip-components=1
bash tools/release/build-mac.sh --check "$TMP/mac/bin" || fail "the Apple Silicon binaries failed build-mac.sh --check"
for b in "$TMP"/mac/bin/*; do
    sig=$(codesign -dvv "$b" 2>&1)
    grep -q '^Authority=Developer ID Application' <<< "$sig" || fail "$(basename "$b") is not signed by a Developer ID (build-mac.sh without MAC_SIGN_IDENTITY)"
done
grep -qF "$(git rev-parse HEAD)" "$TMP/mac/README.md" || fail "$MAC_NAME's README names another commit than HEAD"
echo "publish.sh: $MAC_NAME.tar.gz $(stat -f %z "$TMP/$MAC_NAME.tar.gz") bytes, sha256 $MAC_SHA, checks and Developer ID ok"

# 4. the upload. gh uploads one call's files in parallel, and the image workflow starts when the x86-64 .sha256 is up,
# so that file goes up alone after every other one.
ASSETS=("$NAME" ${GB10_SHA:+"$GB10_NAME"} "$MAC_NAME")
FILES=()
for a in "${ASSETS[@]}"; do FILES+=("$a.tar.gz" "$a.tar.gz.sha256"); done
if [ "$DRY" = 1 ]; then
    echo "publish.sh: dry run: would upload ${ASSETS[*]} (.tar.gz and .sha256) to $TAG and curl every URL"
else
    (cd "$TMP" && gh release upload "$TAG" --repo "$REPO" --clobber "$NAME.tar.gz" "${FILES[@]:2}" \
        && gh release upload "$TAG" --repo "$REPO" --clobber "$NAME.tar.gz.sha256") \
        || fail "gh release upload failed (does release $TAG exist? gh release create $TAG first)"
    for u in "${FILES[@]}"; do
        code=$(curl -fsSI -o /dev/null -w '%{http_code}' "https://github.com/$REPO/releases/download/$TAG/$u" || echo err)
        [ "$code" = 302 ] || [ "$code" = 200 ] || fail "the asset URL answers $code: $u"
    done
    echo "publish.sh: uploaded and every URL resolves"
fi

# 5. the tap
TAPDIR="$HOME/repo/homebrew-tap"
if [ "$TAP" = 1 ]; then
    if [ "$DRY" = 1 ]; then
        echo "publish.sh: dry run: would bump $TAPDIR/Formula/bloomery.rb to $VERSION / $SHA (linux) / ${GB10_SHA:-none} (gb10) / $MAC_SHA (macos) and push"
    else
        [ -d "$TAPDIR/Formula" ] || fail "no tap checkout at $TAPDIR (clone midagedev/homebrew-tap)"
        git -C "$TAPDIR" fetch -q origin && git -C "$TAPDIR" reset -q --hard origin/HEAD
        [ -z "$(git -C "$TAPDIR" status --porcelain)" ] || fail "the tap checkout is dirty"
        F="$TAPDIR/Formula/bloomery.rb"
        python3 - "$F" "$VERSION" "$SHA" "$MAC_SHA" "$GB10_SHA" <<'PY'
import re, sys
f, v, sha, mac_sha, gb10_sha = sys.argv[1:6]
s = open(f).read()
s, n = re.subn(r'version "[^"]*"', f'version "{v}"', s, count=1)
assert n == 1, "no version line"
# Each platform's url and the sha256 on the line after it. The GB10 url is the formula's choice: bumped where it has
# one, and a release without a GB10 archive is refused there, since its url would keep naming an older release.
def bump(s, kind, digest):
    return re.subn(r'(url "[^"]*/releases/download/)v[^/]*/bloomery-[^"/]*-' + re.escape(kind) +
                   r'\.tar\.gz"(\s+)sha256 "[0-9a-f]{64}"',
                   lambda m: f'{m.group(1)}v{v}/bloomery-{v}-{kind}.tar.gz"{m.group(2)}sha256 "{digest}"', s)
for kind, digest in (("linux-x86_64-cuda-sm86", sha), ("macos-arm64-metal", mac_sha)):
    s, n = bump(s, kind, digest)
    assert n == 1, (kind, n)
gb10 = "linux-aarch64-cuda-sm121a"
if gb10_sha:
    s, n = bump(s, gb10, gb10_sha)
    assert n <= 1, (gb10, n)
    if n == 0:
        print(f"publish.sh: the tap's formula has no {gb10} url; Homebrew on a GB10 installs nothing until it has one", file=sys.stderr)
elif gb10 in s:
    sys.exit(f"publish.sh: the tap's formula names a {gb10} archive and this release has none (BLOOMERY_GB10_BOX)")
open(f, 'w').write(s)
PY
        git -C "$TAPDIR" add Formula/bloomery.rb
        git -C "$TAPDIR" commit -q -m "bloomery v$VERSION (linux sha256 $SHA, macos sha256 $MAC_SHA)"
        git -C "$TAPDIR" push -q origin HEAD
        echo "publish.sh: tap bumped to $VERSION"
    fi
fi

# 6. the lead's remainder
echo "publish.sh: left to the lead:"
[ -n "$GB10_SHA" ] || echo "  - no GB10 archive in this release: BLOOMERY_GB10_BOX names the GB10 that ran 'just release-build $VERSION'"
echo "  - release notes: gh release edit $TAG --repo $REPO --notes-file <file>"
echo "  - the ghcr image builds itself from the tag push (.github/workflows/container.yml); watch:"
echo "      gh run watch \$(gh run list --repo $REPO --workflow 'Docker image' --limit 1 --json databaseId -q '.[0].databaseId')"
