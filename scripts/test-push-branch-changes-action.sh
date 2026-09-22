#!/usr/bin/env bash
# Behaviour tests for .github/actions/push-branch-changes (issue #164).
#
# The three pushing workflows now share one commit-and-push script, so that
# script is exercised for real rather than grepped: each case extracts the
# `run:` block from the committed action, runs it against a throwaway git
# repository with a local bare remote standing in for `origin`, and asserts on
# what actually landed there — the commit, the staged paths, the skip, or the
# non-zero exit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ACTION="${1:-$REPO_ROOT/.github/actions/push-branch-changes/action.yml}"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PASSED=0
FAILED=0

if [[ ! -f "$ACTION" ]]; then
  echo "FAIL: composite action not found: $ACTION" >&2
  exit 2
fi

# The action's single `run: |` block, dedented into a runnable script.
PUSH_SCRIPT="$WORK_DIR/push-branch-changes.sh"
awk '
  { line = $0 }
  line ~ /^[[:space:]]*run:[[:space:]]*\|[[:space:]]*$/ && !seen {
    match(line, /^[[:space:]]*/)
    run_indent = RLENGTH
    seen = 1
    next
  }
  !seen { next }
  line ~ /^[[:space:]]*$/ { print ""; next }
  { match(line, /^[[:space:]]*/); indent = RLENGTH }
  indent <= run_indent { seen = 2; nextfile }
  seen == 1 { print substr(line, run_indent + 3) }
' "$ACTION" >"$PUSH_SCRIPT"

if [[ ! -s "$PUSH_SCRIPT" ]]; then
  echo "FAIL: no run: block extracted from $ACTION" >&2
  exit 2
fi
if ! grep -q 'push origin' "$PUSH_SCRIPT"; then
  echo "FAIL: the extracted run: block does not push — extraction is wrong" >&2
  exit 2
fi
chmod +x "$PUSH_SCRIPT"

# new_fixture NAME → a repository at $WORK_DIR/NAME/work whose `origin` is a
# bare repository at $WORK_DIR/NAME/remote.git, both holding one commit on the
# branch `feature`.
new_fixture() {
  local root="$WORK_DIR/$1"
  rm -rf "$root"
  mkdir -p "$root"
  git init --quiet --bare --initial-branch=feature "$root/remote.git"
  git init --quiet --initial-branch=feature "$root/work"
  git -C "$root/work" config user.name "fixture"
  git -C "$root/work" config user.email "fixture@example.invalid"
  printf 'base\n' >"$root/work/tracked.txt"
  printf 'lock\n' >"$root/work/Cargo.lock"
  git -C "$root/work" add tracked.txt Cargo.lock
  git -C "$root/work" commit --quiet -m "base"
  git -C "$root/work" remote add origin "$root/remote.git"
  git -C "$root/work" push --quiet origin feature
  printf '%s' "$root"
}

# run_push ROOT [ENV_ASSIGNMENT...] → exit status in $STATUS, output in
# $OUTPUT. Neither is a command substitution: a subshell would swallow the
# exit status the cases assert on.
STATUS=0
OUTPUT=""
run_push() {
  local root="$1"
  shift
  STATUS=0
  (
    cd "$root/work" && env \
      PR_HEAD_REF="feature" \
      COMMIT_MESSAGE="chore: automated fix" \
      STAGE_PATHS="" \
      GH_PAT="x-token" \
      "$@" bash "$PUSH_SCRIPT"
  ) >"$WORK_DIR/push-output.txt" 2>&1 || STATUS=$?
  OUTPUT="$(cat "$WORK_DIR/push-output.txt")"
}

pass() {
  echo "OK   $1"
  PASSED=$((PASSED + 1))
}

fail() {
  echo "FAIL $1" >&2
  if [[ -n "${2:-}" ]]; then
    printf '%s\n' "$2" >&2
  fi
  FAILED=$((FAILED + 1))
}

remote_head_message() {
  git -C "$1/remote.git" log -1 --format=%s feature
}

remote_files() {
  git -C "$1/remote.git" ls-tree -r --name-only feature | sort | tr '\n' ' '
}

# 1. No paths: every tracked modification is committed and pushed.
root="$(new_fixture commit-all)"
printf 'changed\n' >"$root/work/tracked.txt"
run_push "$root"
if [[ "$STATUS" -ne 0 ]]; then
  fail "commits every tracked change when no paths are given" "$OUTPUT"
elif [[ "$(remote_head_message "$root")" != "chore: automated fix" ]]; then
  fail "commits every tracked change when no paths are given" "$(remote_head_message "$root")"
elif [[ "$(git -C "$root/remote.git" show feature:tracked.txt)" != "changed" ]]; then
  fail "commits every tracked change when no paths are given" "tracked.txt was not pushed"
else
  pass "commits every tracked change when no paths are given"
fi

# 2. Paths given: only those paths are staged, and an untracked file is left
#    behind rather than swept into the bot commit.
root="$(new_fixture commit-paths)"
printf 'moved\n' >"$root/work/Cargo.lock"
printf 'unrelated\n' >"$root/work/tracked.txt"
printf 'scratch\n' >"$root/work/scratch.txt"
run_push "$root" STAGE_PATHS="Cargo.lock"
if [[ "$STATUS" -ne 0 ]]; then
  fail "stages only the paths it was given" "$OUTPUT"
elif [[ "$(git -C "$root/remote.git" show feature:Cargo.lock)" != "moved" ]]; then
  fail "stages only the paths it was given" "Cargo.lock was not pushed"
elif [[ "$(git -C "$root/remote.git" show feature:tracked.txt)" != "base" ]]; then
  fail "stages only the paths it was given" "an unlisted path was committed"
elif [[ "$(remote_files "$root")" != "Cargo.lock tracked.txt " ]]; then
  fail "stages only the paths it was given" "unexpected tree: $(remote_files "$root")"
else
  pass "stages only the paths it was given"
fi

# 3. Several whitespace-separated paths, as the workflows pass them.
root="$(new_fixture commit-many-paths)"
printf 'moved\n' >"$root/work/Cargo.lock"
printf 'changed\n' >"$root/work/tracked.txt"
run_push "$root" STAGE_PATHS="Cargo.lock   tracked.txt"
if [[ "$STATUS" -ne 0 ]]; then
  fail "stages every path in a whitespace-separated list" "$OUTPUT"
elif [[ "$(git -C "$root/remote.git" show feature:tracked.txt)" != "changed" ]]; then
  fail "stages every path in a whitespace-separated list" "the second path was not committed"
else
  pass "stages every path in a whitespace-separated list"
fi

# 4. No credential: refuse loudly, and push nothing.
root="$(new_fixture no-credential)"
printf 'changed\n' >"$root/work/tracked.txt"
run_push "$root" GH_PAT=""
if [[ "$STATUS" -eq 0 ]]; then
  fail "refuses to push with no credential" "expected a non-zero exit"
elif [[ "$OUTPUT" != *"no push credential available"* ]]; then
  fail "refuses to push with no credential" "$OUTPUT"
elif [[ "$(remote_head_message "$root")" != "base" ]]; then
  fail "refuses to push with no credential" "the remote moved"
else
  pass "refuses to push with no credential"
fi

# 5. Head branch deleted while the job ran: skip the push, exit clean.
root="$(new_fixture deleted-branch)"
printf 'changed\n' >"$root/work/tracked.txt"
git -C "$root/remote.git" branch --quiet -D feature 2>/dev/null ||
  git -C "$root/remote.git" update-ref -d refs/heads/feature
run_push "$root"
if [[ "$STATUS" -ne 0 ]]; then
  fail "skips the push when the head branch is gone" "$OUTPUT"
elif [[ "$OUTPUT" != *"no longer exists"* ]]; then
  fail "skips the push when the head branch is gone" "$OUTPUT"
else
  pass "skips the push when the head branch is gone"
fi

# 6. The branch moved while the job ran: rebase, then push both commits.
root="$(new_fixture moved-branch)"
clone="$WORK_DIR/moved-branch/other"
git clone --quiet "$root/remote.git" "$clone"
git -C "$clone" config user.name "other"
git -C "$clone" config user.email "other@example.invalid"
printf 'someone else\n' >"$clone/other.txt"
git -C "$clone" add other.txt
git -C "$clone" commit --quiet -m "chore: another job pushed first"
git -C "$clone" push --quiet origin feature
printf 'changed\n' >"$root/work/tracked.txt"
run_push "$root"
if [[ "$STATUS" -ne 0 ]]; then
  fail "rebases onto a branch that moved, then pushes" "$OUTPUT"
elif [[ "$(remote_head_message "$root")" != "chore: automated fix" ]]; then
  fail "rebases onto a branch that moved, then pushes" "$(remote_head_message "$root")"
elif [[ "$(git -C "$root/remote.git" log --format=%s feature | wc -l | tr -d ' ')" != "3" ]]; then
  fail "rebases onto a branch that moved, then pushes" "the other job's commit was lost"
else
  pass "rebases onto a branch that moved, then pushes"
fi

# 7. A conflicting change on the branch: fail loudly instead of force-pushing
#    over it, and leave no rebase in progress.
root="$(new_fixture rebase-conflict)"
clone="$WORK_DIR/rebase-conflict/other"
git clone --quiet "$root/remote.git" "$clone"
git -C "$clone" config user.name "other"
git -C "$clone" config user.email "other@example.invalid"
printf 'their edit\n' >"$clone/tracked.txt"
git -C "$clone" commit --quiet -am "chore: conflicting edit"
git -C "$clone" push --quiet origin feature
printf 'our edit\n' >"$root/work/tracked.txt"
run_push "$root"
if [[ "$STATUS" -eq 0 ]]; then
  fail "fails loudly on a rebase conflict" "expected a non-zero exit"
elif [[ "$OUTPUT" != *"could not rebase"* ]]; then
  fail "fails loudly on a rebase conflict" "$OUTPUT"
elif [[ "$(remote_head_message "$root")" != "chore: conflicting edit" ]]; then
  fail "fails loudly on a rebase conflict" "the conflicting commit was overwritten"
elif [[ -d "$root/work/.git/rebase-merge" || -d "$root/work/.git/rebase-apply" ]]; then
  fail "fails loudly on a rebase conflict" "a rebase was left in progress"
else
  pass "fails loudly on a rebase conflict"
fi

echo "push-branch-changes action tests: $PASSED passed, $FAILED failed"
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
