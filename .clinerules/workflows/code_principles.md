# Coding Agent Rules — Concise Best Practices

## Purpose
Provide short, actionable rules for automated coding agents to improve code quality, follow project conventions, and produce minimal, reviewable changes.

---

## Core Principles
- **DRY** — Avoid duplication; centralize logic and configuration.
- **KISS** — Prefer the simplest solution that meets requirements.
- **YAGNI** — Do not implement speculative features.
- **Safety-first** — Avoid destructive changes; never expose secrets.

---

## Clean Code Principles
(Consolidated from widely accepted sources: Robert C. Martin, Fowler, SE community.)

### Naming & Intent
- Use **clear, intention-revealing** names based on domain vocabulary.
- Ensure names are consistent, pronounceable, and unambiguous.

### Functions
- Keep functions **small**, focused, and doing **one thing well**.
- Avoid boolean flags; prefer separate functions.
- Keep parameter lists short; group related data into objects.

### Comments
- Code should explain **what**; comments explain **why**.
- Remove obsolete or redundant comments.

### Structure & Formatting
- Use shallow nesting; favor early returns.
- Follow the repository’s formatting and linting rules.
- Minimize vertical and horizontal clutter.

### Objects & Data Structures
- Encapsulate behavior with data; avoid leaking internals.
- Prefer composition over inheritance.

### Error Handling
- Fail fast with meaningful, maintainers-oriented messages.

### Code Smells to Avoid
- Large classes or long functions.
- Duplicate logic, large parameter lists.
- Deep inheritance chains.
- Magic values, clever/unreadable code.

---

## Design & Architecture Guidelines
- Apply SOLID pragmatically (SRP, OCP, LSP, ISP, DIP).
- Prefer readability and maintainability over early optimization.
- Respect existing architectural boundaries and patterns.

---

## Project Conventions & Code Style
- Follow established naming patterns, directory structure, and module organization.
- Use the project’s idiomatic conventions; remain consistent with surrounding code.
- If no conventions exist, apply clear, predictable, idiomatic style for the language.

---

## Functions, Types & Public APIs
- Use explicit, narrow types; avoid overly generic types.
- Keep public APIs predictable, stable, and minimal.
- Provide concise documentation for non-obvious behavior or interfaces.
- Avoid introducing breaking changes unless required by the task.

---

## Dependency Management
- Prefer built-in capabilities and existing repo dependencies.
- Add a dependency only when it materially improves clarity, safety, or productivity.
- When adding one, include a brief justification:
  - Why it’s needed  
  - Alternatives considered  
  - Version, maintenance, and license notes

---

## Change Management & Version Control
- Produce **atomic commits**, each containing one logical change.
- Use commit messages following:  
  **Intent: <summary> — Rationale: <why this change was needed>**
- Explain each change clearly; emphasize the reasoning, not the diff.
- Add short in-code comments when behavior or rationale is non-obvious.
- Submit minimal, scoped PRs with a concise change summary.
- Avoid modifying areas outside the requested scope.

---

## Code Generation & Agent Behavior
- Read and respect project-level files: manifests, configs, lockfiles, build scripts.
- Prefer small, incremental edits over broad refactors unless explicitly requested.
- Ensure changes are reversible, isolated, and easy to audit.
- When uncertain about project conventions or intent, ask rather than assume.

---

## Performance & Security
- Do not prematurely optimize; improve performance only when required by context.
- Never include secrets, credentials, or personal data in code or logs.
- Use safe defaults and validate all external inputs.
