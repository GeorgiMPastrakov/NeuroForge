# Phase 8: Polish and Release

## Goal

Clean the API, documentation, examples, and validation story so NeuroForge feels coherent and finished.

## Ticket 1

Task:
Audit public API.

Context:
Several phases will have added public types.

Goal:
Review names, include paths, namespace usage, ownership, and error messages.

Files likely involved:
Public headers and tests.

Dependencies:
Phases 1 through 7.

Branch:
`polish-release`

Skill/agent:
Code review agent.

Constraints:
Do not perform unrelated rewrites. Preserve working examples.

Expected output:
Clean, consistent public API.

Tests required:
Full build and test suite.

Definition of done:
API is coherent and examples still compile.

Risks:
Late API churn can break docs and examples.

## Ticket 2

Task:
Complete documentation.

Context:
Users need to understand what NeuroForge can and cannot do.

Goal:
Update README and add docs for architecture, tensor, training, autograd, data, serialization, visualization, and examples.

Files likely involved:
`README.md`, `docs/*.md`

Dependencies:
Implemented features.

Branch:
`polish-release`

Skill/agent:
Documentation agent.

Constraints:
Docs must match actual behavior. Do not claim unsupported features.

Expected output:
Documentation explains build, tests, examples, and architecture.

Tests required:
Run documented commands.

Definition of done:
A new user can build, test, and run demos from README alone.

Risks:
Out-of-date docs are worse than sparse docs.

## Ticket 3

Task:
Finalize examples.

Context:
Examples are the proof surface for the framework.

Goal:
Ensure XOR, regression, save/load, and visualizer examples build and run.

Files likely involved:
`examples`, `data`, CMake.

Dependencies:
Phases 4 through 7.

Branch:
`polish-release`

Skill/agent:
Testing and documentation agents.

Constraints:
No hardcoded training results.

Expected output:
Examples demonstrate real framework behavior.

Tests required:
Build all examples. Run XOR and regression examples.

Definition of done:
Examples are reproducible and documented.

Risks:
Long-running examples should not become slow tests.

## Ticket 4

Task:
Run final validation.

Context:
Release needs one reliable command sequence.

Goal:
Run clean configure, build, test, examples, and optional visualizer check.

Files likely involved:
No source files unless validation finds bugs.

Dependencies:
Tickets 1 through 3.

Branch:
`polish-release`

Skill/agent:
Testing agent and code review agent.

Constraints:
Fix only release-blocking issues in this phase.

Expected output:
Final validation log and clean git status.

Tests required:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

Definition of done:
All tests pass, examples run, README commands are true.

Risks:
Skipping final validation will hide integration bugs.

## Ticket 5

Task:
Prepare release summary.

Context:
The project needs a concise final explanation.

Goal:
Write a release summary covering implemented features, limitations, demos, and future work.

Files likely involved:
`docs/release-notes.md` or README section.

Dependencies:
Ticket 4.

Branch:
`polish-release`

Skill/agent:
Documentation agent.

Constraints:
Be accurate and direct.

Expected output:
Release notes explain what is complete and what remains out of scope.

Tests required:
Documentation review only.

Definition of done:
Release summary matches validated behavior.

Risks:
Marketing language can make the project look less credible.
