# Failure Matrix (Generated)

| Example | ANTLR parse | SysIDE compile | Family | Semantic category | First error location |
|---|---|---|---|---|---|
| `01_missing_import_namespace.sysml` | PASS | FAIL | `reference-error` | Name/symbol resolution (context-sensitive) | `7:18` |
| `02_missing_port_type.sysml` | PASS | FAIL | `reference-error` | Name/symbol resolution (context-sensitive) | `8:23` |
| `03_missing_attribute_type.sysml` | PASS | FAIL | `reference-error` | Name/symbol resolution (context-sensitive) | `8:31` |
| `04_missing_specialization_base.sysml` | PASS | FAIL | `reference-error` | Name/symbol resolution (context-sensitive) | `7:21` |
| `05_attribute_definition_nonreferential_feature.sysml` | PASS | FAIL | `reference-error` | Name/symbol resolution (context-sensitive) | `8:20` |
| `06_invocation_not_behavior.sysml` | PASS | FAIL | `invocation-expression-instantiated-type` | Invocation typing (context-sensitive) | `13:17` |
| `07_missing_feature_in_expression.sysml` | PASS | FAIL | `reference-error` | Name/symbol resolution (context-sensitive) | `13:68` |
| `08_missing_root_qualified_namespace.sysml` | PASS | FAIL | `reference-error` | Name/symbol resolution (context-sensitive) | `8:20` |
| `09_redefine_missing_feature.sysml` | PASS | FAIL | `reference-error` | Name/symbol resolution (context-sensitive) | `12:19` |
| `10_missing_namespace_in_type_use.sysml` | PASS | FAIL | `reference-error` | Name/symbol resolution (context-sensitive) | `8:17` |

All rows are explicit counterexamples where grammar acceptance does not imply compilability.