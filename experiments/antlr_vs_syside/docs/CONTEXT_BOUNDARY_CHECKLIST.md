# Context Boundary Checklist: ANTLR vs SysIDE

Use this checklist when explaining why parse success and compile success differ.

## What ANTLR (grammar parsing) can validate
- Token order and keyword placement (`package`, `part def`, `attribute`, `port`, etc.)
- Balanced delimiters and legal production shapes
- Local expression form and operator sequence
- Whether text belongs to the grammar language

## What ANTLR alone cannot validate
- Whether a referenced namespace/type/feature actually exists
- Whether an invocation target is a Behavior
- Whether a redefinition target is inherited and valid
- Whether metamodel constraints are satisfied (e.g., allowed owned feature kinds)

## What SysIDE validates beyond parse
- Name resolution: namespaces, types, and feature chains
- Static typing and classifier conformance
- Invocation typing rules
- Redefinition and specialization consistency
- Structural/metamodel constraints

## Mapping to this experiment
- 10/10 files parse under ANTLR
- 10/10 files fail SysIDE

So each failure is evidence of a **context-sensitive/static-semantic** requirement that is outside grammar membership.
