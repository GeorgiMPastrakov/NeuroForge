# Phase 1: Project Skeleton

## Goal

Create a buildable C++ project foundation with no framework logic yet.

## Ticket 1

Task:
Create the CMake project skeleton.

Context:
The repo has only governance files. NeuroForge needs a build system before implementation work.

Goal:
Add a top-level CMake project that builds a `neuroforge` library and enables tests.

Files likely involved:
`CMakeLists.txt`, `src/CMakeLists.txt`, `tests/CMakeLists.txt`

Dependencies:
None.

Branch:
`scaffold-project`

Skill/agent:
General C++ build agent.

Constraints:
Use C++20 unless toolchain problems require C++17. Do not add external ML dependencies. Do not add comments in code or CMake files.

Expected output:
CMake configures a library target and a test executable.

Tests required:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

Definition of done:
CMake configures, builds, and runs a smoke test.

Risks:
Overbuilding CMake too early will slow all later phases.

## Ticket 2

Task:
Create the required folder structure.

Context:
AGENTS.md defines the long-term module layout.

Goal:
Create `include/neuroforge`, `src`, `tests`, `examples`, `data`, `docs`, and `tools` with module subfolders.

Files likely involved:
Project directories only.

Dependencies:
Ticket 1.

Branch:
`scaffold-project`

Skill/agent:
General project skeleton agent.

Constraints:
Do not add placeholder source files unless needed for CMake.

Expected output:
Folders exist and match the intended architecture.

Tests required:
Verify folder tree and CMake build.

Definition of done:
The repo shape matches AGENTS.md and builds.

Risks:
Empty directories are not tracked by git, so necessary placeholders may be needed only when a directory must be committed.

## Ticket 3

Task:
Add the public umbrella header.

Context:
The final API should use `#include "neuroforge/neuroforge.hpp"`.

Goal:
Add a minimal umbrella header that compiles and provides the `neuroforge` namespace.

Files likely involved:
`include/neuroforge/neuroforge.hpp`

Dependencies:
Ticket 1.

Branch:
`scaffold-project`

Skill/agent:
General C++ API agent.

Constraints:
No framework classes yet.

Expected output:
A test can include the umbrella header.

Tests required:
Smoke test includes the header.

Definition of done:
Header compiles through the test target.

Risks:
Adding fake APIs now will cause later rework.

## Ticket 4

Task:
Add README and smoke test.

Context:
Users need build instructions immediately.

Goal:
Add README with project pitch, build commands, current status, and next phase. Add a minimal smoke test.

Files likely involved:
`README.md`, `tests/test_smoke.cpp`

Dependencies:
Tickets 1 and 3.

Branch:
`scaffold-project`

Skill/agent:
Documentation and testing agent.

Constraints:
README must not claim implemented ML features before they exist.

Expected output:
README is accurate and smoke test proves the include path works.

Tests required:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

Definition of done:
Smoke test passes and README matches actual behavior.

Risks:
Overselling current capability will make the project look fake.
