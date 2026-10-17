# PLAN 05 — Git Commit and Push Workflow

## Rule
After **every** completed upgrade task, do the following in order:

1. `git add <changed files>`
2. `git commit -m "feat: <short description>"` (or `fix:`, `chore:`, `refactor:`)
3. `git push origin main`

## Commit Message Convention
- `feat:` — new feature or page
- `fix:` — bug fix (syntax error, crash, path fix)
- `refactor:` — code restructure without behavior change
- `chore:` — dependency, config, linting, CI
- `docs:` — documentation or plan files

## Branch
- Always work on `main`
- No feature branches unless requested

## Files to Stage
- Only stage the specific files changed for that upgrade
- Do **not** stage `.venv/`, `*.pth`, `*.exe`, `data/`, `models/`, `results/` (gitignored)

## Example Sequence (Phase 1)
```
git add eeg-mi-bci/src/minirocket_engine.py scripts/overtrain_targets.py requirements.txt
git commit -m "fix: stabilize minirocket pipeline save/predict and remove syntax errors"
git push origin main
```

## Pre-Push Checklist
- [ ] Python files compile (`py_compile`)
- [ ] `ruff check` passes on changed files
- [ ] Relevant tests pass (`pytest`)
- [ ] Commit message follows convention
