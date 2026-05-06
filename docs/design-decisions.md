# Design Decisions Log

> Append-only log. Whenever a non-obvious decision is made, add an entry. Format:

```
## [YYYY-MM-DD] <Phase N> — <One-line title>

**Context**: Why this decision came up.

**Options considered**:
1. ...
2. ...

**Decision**: ...

**Reasoning**: ...

**Consequences / things to revisit**: ...
```

---

## [2025-XX-XX] <Phase 0> — Default model choice

**Context**: We need a fast-iteration model for tests and a paper-faithful model for evaluation.

**Decision**: Use `Qwen/Qwen2.5-1.5B-Instruct` for unit tests (fast); use `mistralai/Mistral-7B-Instruct-v0.2` for benchmark evaluation (matches paper).

**Consequences**: Layerwise wrapper must be model-architecture agnostic enough to handle both. Both use RoPE, so this should be fine.

---
