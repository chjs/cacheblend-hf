# Phase N Report

> **Replace `N` with phase number. Save as `reports/phase-N-report.md`.**

## Summary (1-2 sentences)

Brief description of what was achieved this phase.

## What was done

### Implemented

- `src/cacheblend/<module>.py` — what it does, key functions
- `tests/<test>.py` — what it tests

### Tests

| Test | Result | Notes |
|---|---|---|
| `tests/test_X.py::test_Y` | ✅ pass | max_diff = 1.2e-6 |
| `tests/test_X.py::test_Z` | ✅ pass | |

Include numerical results where relevant (max logit diff, F1 score, TTFT, etc.).

### Acceptance criteria checklist

Copy from the corresponding `tasks/phase-N-*.md` and check off:

- [x] Criterion 1
- [x] Criterion 2
- [ ] Criterion 3 — **NOT MET** because ...

## Decisions made

For each non-obvious decision (also append to `docs/design-decisions.md`):

- **Decision**: ...
- **Why**: ...
- **Alternatives considered**: ...

## Deviations from plan

If anything was done differently from `tasks/phase-N-*.md`, explain why.

## Open questions / blockers

Items that need user input before proceeding:

1. Question 1: ...
2. Blocker 1: ... (if any)

## Files changed

```
$ git diff --stat main...HEAD
src/cacheblend/...    | +120 -3
tests/...             | +85 -0
...
```

## Numbers (if applicable)

If the phase produced measurements, put them here as a table or short list.
Example:

| Method | F1 | TTFT (ms) |
|---|---|---|
| full_recompute | 0.31 | 2300 |
| cacheblend (15%) | 0.30 | 700 |

## Next phase prep

What does the next phase need to know?

- Module X exposes function `foo(...)` that Phase N+1 will use.
- Edge case Y is unhandled — Phase N+1 should address.
- Performance target Z is not yet met.

## Suggested next prompt for Claude Code

> Drafting this here makes the user's email→prompt loop faster.

Example:
> "Phase N is done. Now do Phase N+1. Read tasks/phase-N+1-*.md. Notable
> context from Phase N: <key facts>. Pay extra attention to <warning>."
