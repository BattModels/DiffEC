# Dev setup: remotes and machine-to-machine sync

Two-machine workflow (Artemis HPC node + your laptop) with a third
repo in the picture (the original DiffEC paper repo at `BattModels`).
Use this topology on **both** machines.

## Remote topology

```
  ChangwenXu98/DiffEC  ←──── push/pull ────→  your two clones
        (your fork)              (Artemis + laptop)
            ▲
            │ fetch only
            │ (manual sync, never auto)
            │
  BattModels/DiffEC
   (paper source repo)
```

Two remotes, conventional GitHub-fork naming:

| Name | URL | Direction |
|---|---|---|
| `origin` | `git@github.com:ChangwenXu98/DiffEC.git` | push + fetch |
| `upstream` | `git@github.com:BattModels/DiffEC.git` | fetch only |

Use SSH URLs (not HTTPS) so push works without an interactive password
prompt — Artemis and most laptops have an SSH key configured for
GitHub. Verify with `ssh -T git@github.com`.

Branch tracking:

| Branch | Tracks |
|---|---|
| `feat/tb-sci-task` | `origin/feat/tb-sci-task` (your fork) |
| `main` | `origin/main` (your fork) |

This means `git push` / `git pull` on `feat/tb-sci-task` always
flows through your fork — you can't accidentally push the TB-Sci
work to `BattModels/DiffEC`.

## One-time setup on each machine

If a clone has wrong remote names or HTTPS URLs:

```bash
# Replace any existing 'origin' that points at BattModels:
git remote remove origin 2>/dev/null
git remote remove upstream 2>/dev/null
git remote add origin   git@github.com:ChangwenXu98/DiffEC.git
git remote add upstream git@github.com:BattModels/DiffEC.git

git fetch --all --prune
git branch -u origin/feat/tb-sci-task feat/tb-sci-task
git branch -u origin/main main

# Sanity:
git remote -v
git for-each-ref --format='%(refname:short) -> %(upstream:short)' refs/heads/
```

A fresh clone of your fork already has `origin` correct; you only need
to add `upstream` and (optionally) point `main` at it for sync:

```bash
git clone git@github.com:ChangwenXu98/DiffEC.git
cd DiffEC
git remote add upstream git@github.com:BattModels/DiffEC.git
git fetch upstream
```

## Day-to-day sync between Artemis and laptop

Both machines push/pull through `origin` (your fork). Standard rhythm:

```bash
# After finishing work on machine A:
git push

# On machine B, before starting work:
git pull --ff-only
```

If both machines have made commits and diverged on the same branch,
`git pull --ff-only` will refuse rather than do a surprise merge.
Resolve explicitly:

```bash
git fetch
git log --oneline HEAD..origin/feat/tb-sci-task        # what they did
git log --oneline origin/feat/tb-sci-task..HEAD        # what you did
# Pick one of:
git rebase origin/feat/tb-sci-task                     # replay your commits on top of theirs (preferred for clean history)
git merge  origin/feat/tb-sci-task                     # merge commit (preserves both lines verbatim)
git push --force-with-lease                            # only after rebase, only if you're sure your local is correct
```

`--force-with-lease` (not `--force`) refuses to overwrite if someone
else pushed in the meantime — a small safety net.

## Pulling updates from BattModels (the paper repo)

Rarely needed for the TB-Sci work (we develop on `feat/tb-sci-task`,
not on `main`). But if you ever want to refresh your fork's `main`:

```bash
git fetch upstream
git checkout main
git merge --ff-only upstream/main   # refuses if your main diverged
git push                            # pushes to your fork's main
```

Never push to `upstream` (BattModels). Pushing to `origin` (your fork)
is fine — it's yours.

## Where the TB-Sci PR eventually goes

The final destination is a SEPARATE repo, not BattModels:

- `harbor-framework/terminal-bench-science` (the TB-Sci benchmark repo)

That submission is a one-time copy of just
`tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport/`
into a PR there. It is **not** a git-merge relationship with this repo
— ADR-0009 spells out how that transfer works.

So during day-to-day development, you only ever interact with
`origin` (your fork). `upstream` is fetch-only and rarely touched.
