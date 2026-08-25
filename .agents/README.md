# Cross-runtime coordination

This directory is the vendor-neutral coordination substrate for agent runtimes
and human contributors.

## Required workflow

1. Create an exclusively owned worktree and runtime-labelled branch.
2. Add a pending task JSON under `.agents/tasks/`.
3. Run `agentic-task claim <repo>` from a clean worktree.
4. Edit only paths owned by the active task.
5. Verify the change, integrate it through Git, then run
   `agentic-task complete <repo> <task-id>`.

Git push is the cross-machine arbitration point. If push does not confirm
success, the CLI keeps the local metadata commit and exits non-zero. Reconcile
explicitly; never reset or rebase a caller's working tree automatically.

`schema.json` defines the task format. `install-manifest.json` records only the
artifacts managed or adopted by the installer so rollback is narrow and
drift-safe.
