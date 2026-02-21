# Error Family Breakdown (Generated)

- Total files: `10`
- ANTLR parse pass: `10`
- SysIDE compile fail: `10`

## Family Counts

| Error family | Count | Share |
|---|---:|---:|
| `reference-error` | 9 | 90.0% |
| `invocation-expression-instantiated-type` | 1 | 10.0% |

## Interpretation

Most failures are `reference-error`, which is expected: ANTLR accepts identifier forms syntactically, while SysIDE requires successful symbol resolution.
Other families (if present) reflect additional context-sensitive static rules enforced post-parse.