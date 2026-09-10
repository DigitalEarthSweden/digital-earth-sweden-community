<!-- agentic-task:coordination:start -->
## Cross-runtime coordination mechanics

Shared policy lives in `AGENTS.md`. Claude-specific hooks may enforce it but
must not weaken or duplicate that policy. Use a Claude worktree for every
writing session and the vendor-neutral `agentic-task` CLI for task claims.
Use `~/.agents/bin/agentic-continuity` for personal checkpoints, live messages,
and active-job locks; commit only validated task-scoped `.agents/handoffs/`
records for another machine.
<!-- agentic-task:coordination:end -->
