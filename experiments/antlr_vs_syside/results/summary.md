# ANTLR vs SysIDE Summary

- Total examples: 10
- ANTLR parse pass: 10
- SysIDE compile pass: 0
- Mismatch count (parse PASS, compile FAIL): 10

## Mismatch Examples
- `01_missing_import_namespace.sysml`
- `02_missing_port_type.sysml`
- `03_missing_attribute_type.sysml`
- `04_missing_specialization_base.sysml`
- `05_attribute_definition_nonreferential_feature.sysml`
- `06_invocation_not_behavior.sysml`
- `07_missing_feature_in_expression.sysml`
- `08_missing_root_qualified_namespace.sysml`
- `09_redefine_missing_feature.sysml`
- `10_missing_namespace_in_type_use.sysml`

## Highlighted Diagnostics
### 01_missing_import_namespace.sysml
```text
SYSIDE_COMPILE_FAIL /Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/01_missing_import_namespace.sysml
/Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/01_missing_import_namespace.sysml:7:18: error (reference-error): No Namespace named 'MissingPkg01' found.
    7 |   private import MissingPkg01::*;
      |                  ^^^^^^^^^^^^
```

### 02_missing_port_type.sysml
```text
SYSIDE_COMPILE_FAIL /Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/02_missing_port_type.sysml
/Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/02_missing_port_type.sysml:8:23: error (reference-error): No Type named 'MissingPortType02' found.
    8 |     port controlPort: MissingPortType02;
      |                       ^^^^^^^^^^^^^^^^^
```
