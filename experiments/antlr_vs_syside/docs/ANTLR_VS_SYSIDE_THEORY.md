# ANTLR vs SysIDE: Formal Boundary

## Executive claim
`ANTLR parse pass` means the text belongs to the grammar language `L(G)` for some grammar `G`.
`SysIDE compile pass` means more than syntax: it requires additional semantic/static constraints over the parsed model.

In this repository, ANTLR parse outcomes are produced by the third-party HAMR parser implementation (pinned in `experiments/antlr_vs_syside/third_party/hamr-sysml-parser`), not by a locally-authored grammar.

So the accepted set for SysIDE is:

`C = { x in L(G) | P1(x) and P2(x) and ... and Pk(x) }`

where each `Pi` is a semantic/toolchain predicate such as name resolution, typing, specialization conformance, or metamodel rules.

This implies `C` is a strict subset of `L(G)` when at least one file parses but fails SysIDE.

## Why ANTLR is context-free here
ANTLR grammars are grammar formalisms used to define token/production structure. They can enforce local syntactic forms and precedence, but they do not by themselves enforce global symbol-table properties like:
- "this referenced namespace must exist"
- "this invoked element must be a Behavior"
- "this feature redefinition target must be declared in the inherited context"

Those checks depend on model-wide context and cross-reference resolution.

## Why SysIDE is not "just context-free parsing"
`syside check` runs a full static pipeline after parse, including (at minimum):
- namespace and symbol resolution
- type/classifier conformance
- feature and ownership constraints
- invocation validity constraints

These checks are context-sensitive with respect to the model graph and cannot be reduced to plain CFG membership.

## Concrete evidence from this experiment
In this folder we have 10 files where:
- ANTLR parse: 10/10 pass
- SysIDE compile: 0/10 pass

See:
- `docs/FAILURE_CATALOG.generated.md`
- `docs/ERROR_FAMILY_BREAKDOWN.generated.md`
- `results/results.csv`

## Minimal counterexample argument
If there exists any file `x` such that:
- `x in L(G)` (ANTLR pass), and
- `x notin C` (SysIDE fail),
then grammar membership is insufficient for compilability.

This experiment provides 10 such files, so the claim is empirically established for this setup.
