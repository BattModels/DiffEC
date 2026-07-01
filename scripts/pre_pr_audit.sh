#!/bin/bash
# Pre-PR audit for the tasks/.../concentrated-electrolyte-mass-transport
# subtree that ships to harbor-framework/terminal-bench-science.
#
# Runs 14 checks. Prints one line per check with PASS/FAIL and a short
# summary. Exits 0 iff all pass. Safe to re-run any time; no side
# effects on the repo (read-only).
#
# Usage:
#     bash scripts/pre_pr_audit.sh
#
# See docs/pre-pr-checklist.md for the narrative wrapper (when to
# run, what to do on failure, revert steps, PR submission).

set -uo pipefail

TASK="tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport"
FAIL=0

pass() { printf "  ✓ %s\n" "$1"; }
fail() { printf "  ✗ %s\n"    "$1"; FAIL=$((FAIL + 1)); }
note() { printf "     %s\n" "$1"; }
hdr()  { printf "\n[%2d] %s\n" "$1" "$2"; }

[ -d "$TASK" ] || { echo "ERROR: run from repo root; $TASK missing"; exit 2; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || { echo "ERROR: not inside a git work tree"; exit 2; }

echo "==========================================================="
echo "Pre-PR audit for $TASK"
echo "==========================================================="

# ---------------------------------------------------------------------
hdr 1 "Complete subtree inventory (files git tracks in $TASK)"
n_files=$(git ls-files "$TASK" | wc -l)
subtree_size=$(du -sk "$TASK" | awk '{print $1}')
note "$n_files tracked files, $subtree_size KB total on disk"
[ "$n_files" -ge 20 ] && [ "$n_files" -le 60 ] && pass "file count in expected range 20–60" \
    || fail "file count $n_files out of expected range 20–60"

# ---------------------------------------------------------------------
hdr 2 "No file exceeds 100 MB (upstream policy limit)"
oversized=$(find "$TASK" -type f -size +100M 2>/dev/null)
[ -z "$oversized" ] && pass "no files > 100 MB" || { fail "oversized files found"; echo "$oversized" | while read f; do note "$f"; done; }

# ---------------------------------------------------------------------
hdr 3 "Agent view (environment/) leaks nothing sensitive"
# Files under environment/ that git tracks:
agent_files=$(git ls-files "$TASK/environment")
n_agent=$(echo "$agent_files" | wc -l)
note "$n_agent files in environment/"
# Must NOT contain oracle_truth or oracle/ code
leak=$(echo "$agent_files" | grep -E "oracle_truth|/oracle/|truth\.npz")
[ -z "$leak" ] && pass "no oracle_truth or oracle/ code in agent view" \
    || { fail "leaked truth/oracle files in environment/"; echo "$leak" | while read f; do note "$f"; done; }

# ---------------------------------------------------------------------
hdr 4 "Verifier (tests/) contains oracle + truth"
for req in "tests/oracle/solver.py" "tests/oracle/flux.py" "tests/oracle/invert_ne.py" \
           "tests/test_outputs.py" "tests/test.sh" "tests/Dockerfile"; do
    if git ls-files "$TASK/$req" | grep -q .; then pass "$req present"; else fail "$req MISSING"; fi
done
for i in 1 2 3 4; do
    if git ls-files "$TASK/tests/oracle_truth/case_$i/truth.npz" | grep -q .; then
        pass "tests/oracle_truth/case_$i/truth.npz present"
    else
        fail "tests/oracle_truth/case_$i/truth.npz MISSING"
    fi
done

# ---------------------------------------------------------------------
hdr 5 "Reference solution (solution/) has entrypoint + helpers"
for req in "solution/solve.sh" "solution/reference_solver.py" "solution/pde.py" \
           "solution/parameterize.py" "solution/lab_frame_solver.py"; do
    if git ls-files "$TASK/$req" | grep -q .; then pass "$req present"; else fail "$req MISSING"; fi
done

# ---------------------------------------------------------------------
hdr 6 "No untracked non-ignored files inside subtree"
stray=$(cd "$TASK" && git status --short --untracked-files=normal .)
[ -z "$stray" ] && pass "no stray untracked files" \
    || { fail "stray files found (would ship if added)"; echo "$stray" | head -10 | while read l; do note "$l"; done; }

# ---------------------------------------------------------------------
hdr 7 "formalism.md doesn't leak oracle numbers"
# 'oracle' should appear only in spec-of-check context, not as a literal value
oracle_hits=$(grep -c "oracle" "$TASK/environment/data/cases/case_1/formalism.md")
note "'oracle' mentions in case_1 formalism.md: $oracle_hits (all should be spec-of-check)"
# Must NOT contain literal D_table_cgs or tp0_table values from case_gen configs
leaked_tp0=$(grep -E "0\.4[12]\s*,\s*0\.3[78]" "$TASK/environment/data/cases/case_1/formalism.md")
[ -z "$leaked_tp0" ] && pass "no tp0_table-like literals in formalism.md" \
    || fail "possible leaked tp0 values in formalism.md"

# ---------------------------------------------------------------------
hdr 8 "PILOT-ONLY docker_image lines must be REVERTED before PR"
if grep -q "^docker_image" "$TASK/task.toml"; then
    fail "task.toml still contains 'docker_image' line(s) — must remove before PR"
    grep -n "^docker_image\|PILOT-ONLY" "$TASK/task.toml" | while read l; do note "$l"; done
elif grep -q "^\[verifier\.environment\]" "$TASK/task.toml"; then
    fail "task.toml still contains [verifier.environment] section — must remove before PR"
else
    pass "task.toml has no PILOT-ONLY docker_image lines"
fi

# ---------------------------------------------------------------------
hdr 9 "task.toml validates against Harbor 0.16 schema"
HARBOR_PY="$HOME/.local/share/uv/tools/harbor/bin/python"
if [ -x "$HARBOR_PY" ]; then
    if PYTHONPATH="$HOME/.local/share/uv/tools/harbor/lib/python3.13/site-packages" \
        "$HARBOR_PY" -c "
import warnings; warnings.simplefilter('ignore')
from harbor.models.task.config import TaskConfig
cfg = TaskConfig.model_validate_toml(open('$TASK/task.toml').read())
print(f'schema_version={cfg.schema_version}, artifacts={len(cfg.artifacts)}')
" 2>/dev/null; then
        pass "task.toml parses cleanly under Harbor 0.16 schema"
    else
        fail "task.toml FAILS to parse under Harbor 0.16 schema"
    fi
else
    note "harbor CLI not installed; skipping (uv tool install harbor)"
fi

# ---------------------------------------------------------------------
hdr 10 "Cross-refs: solve.sh writes to task.toml's artifacts paths"
if grep -q '/root/results' "$TASK/solution/solve.sh" \
   && grep -q '/root/results/case_.*/transport.json' "$TASK/task.toml"; then
    pass "solve.sh output path matches task.toml artifacts"
else
    fail "path mismatch between solve.sh and task.toml artifacts"
fi
if grep -q '/logs/verifier/reward.txt' "$TASK/tests/test.sh"; then
    pass "test.sh writes /logs/verifier/reward.txt (Harbor convention)"
else
    fail "test.sh missing reward.txt write"
fi

# ---------------------------------------------------------------------
hdr 11 "reference_solver.py CLI matches solve.sh's invocation"
if grep -q "reference_solver.py" "$TASK/solution/solve.sh" \
   && grep -q -- "--cases" "$TASK/solution/solve.sh" \
   && grep -q -- "--out"    "$TASK/solution/solve.sh"; then
    pass "solve.sh invokes reference_solver.py with --cases and --out"
else
    fail "solve.sh invocation missing reference_solver.py or --cases/--out"
fi
if grep -q "add_argument.*--cases" "$TASK/solution/reference_solver.py" \
   && grep -q "add_argument.*--out"    "$TASK/solution/reference_solver.py"; then
    pass "reference_solver.py accepts --cases and --out"
else
    fail "reference_solver.py CLI doesn't match"
fi

# ---------------------------------------------------------------------
hdr 12 "Dockerfiles have pinned versions (no ':latest')"
for df in "$TASK/environment/Dockerfile" "$TASK/tests/Dockerfile"; do
    latest=$(grep -E "FROM.*:latest|==\s*latest" "$df" || true)
    if [ -z "$latest" ]; then
        pass "$(basename $(dirname $df))/Dockerfile — no :latest tags"
    else
        fail "$(basename $(dirname $df))/Dockerfile uses unpinned :latest"
    fi
done

# ---------------------------------------------------------------------
hdr 13 "No hardcoded local/HPC paths in shipped files"
paths=$(git ls-files "$TASK" | xargs grep -l "/nfs/turbo\|/scratch/venkvis\|/Users/\|/home/[a-z]\|artemis" 2>/dev/null)
[ -z "$paths" ] && pass "no hardcoded HPC/laptop paths" \
    || { fail "hardcoded paths found"; echo "$paths" | while read f; do note "$f"; done; }

# ---------------------------------------------------------------------
hdr 14 "No TODO / FIXME / XXX in shipped files"
todos=$(git ls-files "$TASK" | xargs grep -Hn "TODO\|FIXME\|XXX" 2>/dev/null | grep -v "^Binary")
[ -z "$todos" ] && pass "no unresolved TODO/FIXME/XXX" \
    || { fail "TODOs found"; echo "$todos" | head -5 | while read l; do note "$l"; done; }

# ---------------------------------------------------------------------
echo
echo "==========================================================="
if [ "$FAIL" -eq 0 ]; then
    echo "AUDIT PASSED — subtree is ready for the upstream PR."
    exit 0
else
    echo "AUDIT FAILED — $FAIL check(s) blocked. Address before PR."
    exit 1
fi
