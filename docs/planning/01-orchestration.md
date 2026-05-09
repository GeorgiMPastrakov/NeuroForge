# NeuroForge Orchestration Plan

## Branch Policy

Use short descriptive branch names:

```text
scaffold-project
math-core-tensor
nn-loss-optim
training-xor
autograd-engine
data-metrics-serialization
visual-lab
polish-release
```

Never use branch names with the forbidden Codex prefix from `AGENTS.md`.

## Commit Policy

Commit after each meaningful working change.

Use short messages:

```text
build: add project skeleton
feat: add tensor matmul
test: add tensor shape tests
feat: add sequential model
feat: train xor demo
docs: add architecture guide
```

Do not commit broken code unless explicitly requested.

## Worktree Policy

Use one worktree until Phase 1 is merged.

Use worktrees only when tasks do not edit the same files.

Safe later splits:

- `nn-loss-optim`: split layers and optimizer only after Tensor API is stable
- `data-metrics-serialization`: split CSV/metrics/save-load if public APIs are already merged
- `visual-lab`: can run mostly alone after `TrainingHistory` and model summary exist

Unsafe splits:

- Tensor and layers at the same time before Tensor API exists
- Trainer and loss API at the same time before loss backward is decided
- Visual Lab before real `TrainingHistory` exists

## Merge Order

Merge only through this order:

```text
main
  <- scaffold-project
  <- math-core-tensor
  <- nn-loss-optim
  <- training-xor
  <- autograd-engine
  <- data-metrics-serialization
  <- visual-lab
  <- polish-release
```

If a later branch starts early, rebase it after each dependency merges.

## Standard Dispatch Prompt

```text
Use $SKILL_NAME.

Task:

Context:

Goal:

Files likely involved:

Dependencies:

Branch:

Constraints:
- No comments in C++ code.
- No external ML libraries for framework logic.
- No branch names with the forbidden Codex prefix from AGENTS.md.

Expected output:

Tests required:

Definition of done:

Risks:
```

## Review Rules

Before merge, check:

- public API matches the phase plan
- all expected tests exist
- shape and input errors are clear
- no generated junk is committed
- no comments were added to C++ code
- no unrelated refactors are included
- CMake, build, and test commands pass

## Required Local Checks

For C++ implementation changes:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

For planning docs:

```bash
find docs/planning -maxdepth 1 -type f | sort
grep the planning docs for the forbidden branch prefix from AGENTS.md
```

The prefix check should produce no matches.

## Risk Gates

Do not start Phase 3 until:

- `Tensor` has tested shape validation
- `Tensor::matmul` works
- activations exist or are planned in Math Core

Do not start Phase 4 until:

- `Sequential` forwards correctly
- `Linear` exposes parameters
- `MSELoss` has backward output
- `SGD` updates parameters

Do not start Phase 7 until:

- `TrainingHistory` stores loss per epoch
- model summary exposes real layer metadata
- at least one demo produces real training data
