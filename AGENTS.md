<!-- agentic-task:coordination:start -->
## Cross-runtime coordination

This section is managed by `agentic-task`. Repository-specific instructions
outside this block remain authoritative.

- Every writing agent session uses an exclusively owned worktree.
- Use runtime-labelled branches: `agent/<initials>/<runtime>/<area>-<slug>`.
- Claim a task before editing and complete it afterward. Claims arbitrate whole
  tasks; coordinate intended file scope explicitly because the schema does not
  enforce path ownership.
- Transfer durable execution state through immutable, task-scoped files under
  `.agents/handoffs/<task-id>/`; never use one mutable `LATEST.md`.
- Never share a branch and working tree between concurrent writing sessions.
- A dirty working tree belongs to its current session; other agents must not
  edit it.
- Questions and diagnosis do not authorize mutation. Verify non-trivial work
  before declaring it complete.
- Personal checkpoints, snapshots, live inbox messages, active-session state,
  and running-code locks use the shared local `~/.agents/continuity/` store
  through runtime adapters, not vendor stores.
<!-- agentic-task:coordination:end -->
