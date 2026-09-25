# Cross-runtime coordination

This directory is the vendor-neutral coordination substrate for agent runtimes
and human contributors.

## Required workflow

1. Create an exclusively owned worktree and runtime-labelled branch.
2. Add and commit a `pending` task JSON under `.agents/tasks/`. For a remote
   repository, first obtain its credential-free identity and store the result
   as `coordination_authority`:

   ```bash
   agentic-task authority . --remote origin
   ```

   Relative URLs, embedded credentials, URL rewrite rules, and differing fetch
   and push targets are rejected.
3. From the clean worktree, claim it against the canonical remote:

   ```bash
   AGENT_ID=te-codex AGENT_RUNTIME=codex \
     agentic-task claim . --remote origin
   ```

   Omit `--remote` only when the repository genuinely has no remotes.
4. Coordinate the task's intended file scope explicitly. A claim arbitrates
   the whole task; this schema does not infer or enforce path ownership.
5. When ownership moves to another runtime or machine, write an immutable
   record under `.agents/handoffs/<task-id>/` with
   `~/.agents/bin/agentic-continuity handoff`. Validate it against
   `.agents/handoff-schema.json` and commit it
   with the task branch. Never update a shared `LATEST.md` pointer.
6. Verify and integrate the change, then complete the task with the same
   identity, runtime, and canonical remote:

   ```bash
   AGENT_ID=te-codex AGENT_RUNTIME=codex \
     agentic-task complete . <task-id> --remote origin
   ```

## Claim authority and recovery

For repositories with a remote, the authoritative lock is the deterministic
`refs/heads/agentic-task/claims/<task-id>` ref. The CLI updates it with an exact
compare-and-swap and confirms its SHA after every push. Different agent branches
therefore cannot both acquire the same task. The hashed
`coordination_authority` prevents two forks or differently
configured remotes from silently creating independent locks without committing
transport credentials or private URL material.

If publication is rejected or ambiguous, the CLI keeps the local metadata
commit and exits non-zero. Inspect the reported commit, the authoritative ref,
and the task owner, then reconcile explicitly. Never reset, rebase, or rewrite
a caller's working tree automatically.

The operational lifecycle for this profile is `pending → claimed → completed`.
The CLI verifies owner, runtime, and authoritative-ref ancestry on completion.
Other schema statuses are reserved for governance workflows and must not be
hand-edited into this coordination profile.

`schema.json` defines the task format; `handoff-schema.json` defines the
committable cross-runtime handoff. `install-manifest.json` records only the
artifacts managed or adopted by the installer so rollback is narrow and
drift-safe. Personal checkpoints, mechanical snapshots, the live inbox,
active-session state, and active-job locks stay outside Git under
`~/.agents/continuity/`.
