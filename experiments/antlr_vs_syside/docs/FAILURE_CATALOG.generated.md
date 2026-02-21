# Failure Catalog (Generated)

This file is generated from `results/results.csv`.
Each entry documents exactly why SysIDE rejects a file that ANTLR accepts.

## 1. `01_missing_import_namespace.sysml`

- `parse_ok`: `True`
- `compile_ok`: `False`
- `primary_error_family`: `reference-error`
- `first_error_location`: `line 7, col 18`
- `first_error_message`: No Namespace named 'MissingPkg01' found.

Source context:
```text
   5 | // Expected failure focus: import line
   6 | package Distinct01 {
>  7 |   private import MissingPkg01::*;
   8 |   part def A {}
   9 | }
```

SysIDE diagnostic snippet:
```text
SYSIDE_COMPILE_FAIL /Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/01_missing_import_namespace.sysml
/Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/01_missing_import_namespace.sysml:7:18: error (reference-error): No Namespace named 'MissingPkg01' found.
    7 |   private import MissingPkg01::*;
```

## 2. `02_missing_port_type.sysml`

- `parse_ok`: `True`
- `compile_ok`: `False`
- `primary_error_family`: `reference-error`
- `first_error_location`: `line 8, col 23`
- `first_error_message`: No Type named 'MissingPortType02' found.

Source context:
```text
   6 | package Distinct02 {
   7 |   part def Controller {
>  8 |     port controlPort: MissingPortType02;
   9 |   }
  10 | }
```

SysIDE diagnostic snippet:
```text
SYSIDE_COMPILE_FAIL /Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/02_missing_port_type.sysml
/Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/02_missing_port_type.sysml:8:23: error (reference-error): No Type named 'MissingPortType02' found.
    8 |     port controlPort: MissingPortType02;
```

## 3. `03_missing_attribute_type.sysml`

- `parse_ok`: `True`
- `compile_ok`: `False`
- `primary_error_family`: `reference-error`
- `first_error_location`: `line 8, col 31`
- `first_error_message`: No Type named 'MissingVoltageType03' found.

Source context:
```text
   6 | package Distinct03 {
   7 |   part def Battery {
>  8 |     attribute nominalVoltage: MissingVoltageType03;
   9 |   }
  10 | }
```

SysIDE diagnostic snippet:
```text
SYSIDE_COMPILE_FAIL /Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/03_missing_attribute_type.sysml
/Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/03_missing_attribute_type.sysml:8:31: error (reference-error): No Type named 'MissingVoltageType03' found.
    8 |     attribute nominalVoltage: MissingVoltageType03;
```

## 4. `04_missing_specialization_base.sysml`

- `parse_ok`: `True`
- `compile_ok`: `False`
- `primary_error_family`: `reference-error`
- `first_error_location`: `line 7, col 21`
- `first_error_message`: No Classifier named 'MissingBaseMotor04' found.

Source context:
```text
   5 | // Expected failure focus: specialization target
   6 | package Distinct04 {
>  7 |   part def Motor :> MissingBaseMotor04 {
   8 |   }
   9 | }
```

SysIDE diagnostic snippet:
```text
SYSIDE_COMPILE_FAIL /Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/04_missing_specialization_base.sysml
/Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/04_missing_specialization_base.sysml:7:21: error (reference-error): No Classifier named 'MissingBaseMotor04' found.
    7 |   part def Motor :> MissingBaseMotor04 {
```

## 5. `05_attribute_definition_nonreferential_feature.sysml`

- `parse_ok`: `True`
- `compile_ok`: `False`
- `primary_error_family`: `reference-error`
- `first_error_location`: `line 8, col 20`
- `first_error_message`: No Type named 'MissingBehaviorType05' found.

Source context:
```text
   6 | package Distinct05 {
   7 |   part def OperatorConsole {
>  8 |     action start : MissingBehaviorType05;
   9 |   }
  10 | }
```

SysIDE diagnostic snippet:
```text
SYSIDE_COMPILE_FAIL /Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/05_attribute_definition_nonreferential_feature.sysml
/Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/05_attribute_definition_nonreferential_feature.sysml:8:12: warning (namespace-distinguishability): Member name 'start' shadows Parts::Part::start
│   8 |     action start : MissingBehaviorType05;
```

## 6. `06_invocation_not_behavior.sysml`

- `parse_ok`: `True`
- `compile_ok`: `False`
- `primary_error_family`: `invocation-expression-instantiated-type`
- `first_error_location`: `line 13, col 17`
- `first_error_message`: Invocation expression must invoke a `Behavior` or a feature typed by a single `Behavior`

Source context:
```text
  11 |   part camera {
  12 |     action takePicture {
> 13 |       then send Show() via displayPort;
  14 |     }
  15 |     port displayPort;
```

SysIDE diagnostic snippet:
```text
SYSIDE_COMPILE_FAIL /Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/06_invocation_not_behavior.sysml
/Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/06_invocation_not_behavior.sysml:13:17: error (invocation-expression-instantiated-type): Invocation expression must invoke a `Behavior` or a feature typed by a single `Behavior`
    13 |       then send Show() via displayPort;
```

## 7. `07_missing_feature_in_expression.sysml`

- `parse_ok`: `True`
- `compile_ok`: `False`
- `primary_error_family`: `reference-error`
- `first_error_location`: `line 13, col 68`
- `first_error_message`: No Feature named 'height' found.

Source context:
```text
  11 |   part def Wheel {
  12 |     attribute hubDiameter: LengthValue;
> 13 |     attribute outerDiameter: LengthValue = (hubDiameter + 2 * tire.height);
  14 |     part tire: Tire[1];
  15 |   }
```

SysIDE diagnostic snippet:
```text
SYSIDE_COMPILE_FAIL /Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/07_missing_feature_in_expression.sysml
/Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/07_missing_feature_in_expression.sysml:13:68: error (reference-error): No Feature named 'height' found.
    13 |     attribute outerDiameter: LengthValue = (hubDiameter + 2 * tire.height);
```

## 8. `08_missing_root_qualified_namespace.sysml`

- `parse_ok`: `True`
- `compile_ok`: `False`
- `primary_error_family`: `reference-error`
- `first_error_location`: `line 8, col 20`
- `first_error_message`: No Namespace named 'NoSuchRoot08' found.

Source context:
```text
   6 | package Distinct08 {
   7 |   part def Controller {
>  8 |     port comms: $::NoSuchRoot08::BusType;
   9 |   }
  10 | }
```

SysIDE diagnostic snippet:
```text
SYSIDE_COMPILE_FAIL /Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/08_missing_root_qualified_namespace.sysml
/Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/08_missing_root_qualified_namespace.sysml:8:20: error (reference-error): No Namespace named 'NoSuchRoot08' found.
    8 |     port comms: $::NoSuchRoot08::BusType;
```

## 9. `09_redefine_missing_feature.sysml`

- `parse_ok`: `True`
- `compile_ok`: `False`
- `primary_error_family`: `reference-error`
- `first_error_location`: `line 12, col 19`
- `first_error_message`: No Feature named 'missing' found.

Source context:
```text
  10 |   }
  11 |   part def Derived :> Base {
> 12 |     attribute :>> missing = 1;
  13 |   }
  14 | }
```

SysIDE diagnostic snippet:
```text
SYSIDE_COMPILE_FAIL /Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/09_redefine_missing_feature.sysml
/Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/09_redefine_missing_feature.sysml:12:19: error (reference-error): No Feature named 'missing' found.
    12 |     attribute :>> missing = 1;
```

## 10. `10_missing_namespace_in_type_use.sysml`

- `parse_ok`: `True`
- `compile_ok`: `False`
- `primary_error_family`: `reference-error`
- `first_error_location`: `line 8, col 17`
- `first_error_message`: No Namespace named 'MissingNS10' found.

Source context:
```text
   6 | package Distinct10 {
   7 |   part def Vehicle {
>  8 |     part wheel: MissingNS10::Wheel;
   9 |   }
  10 | }
```

SysIDE diagnostic snippet:
```text
SYSIDE_COMPILE_FAIL /Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/10_missing_namespace_in_type_use.sysml
/Users/chancelavoie/Desktop/AI_AGENT_Benchmarking_Compiler_in_Loop/experiments/antlr_vs_syside/examples/mismatch_10_distinct/10_missing_namespace_in_type_use.sysml:8:17: error (reference-error): No Namespace named 'MissingNS10' found.
    8 |     part wheel: MissingNS10::Wheel;
```
