# Why Each Example Fails (Human-readable)

This document explains the intent behind each distinct mismatch case.
Each file is designed to parse as valid SysML text shape while violating a semantic/toolchain rule checked by SysIDE.

## 01 `01_missing_import_namespace.sysml`
- Failure focus: import target namespace
- Why it fails: `private import MissingPkg01::*;` references a namespace that does not exist.
- Typical SysIDE family: `reference-error`

## 02 `02_missing_port_type.sysml`
- Failure focus: port type lookup
- Why it fails: `port controlPort: MissingPortType02;` uses an undefined type.
- Typical family: `reference-error`

## 03 `03_missing_attribute_type.sysml`
- Failure focus: attribute type lookup
- Why it fails: attribute type name is undefined.
- Typical family: `reference-error`

## 04 `04_missing_specialization_base.sysml`
- Failure focus: specialization target (`:>`)
- Why it fails: specialized base classifier is undefined.
- Typical family: `reference-error`

## 05 `05_attribute_definition_nonreferential_feature.sysml`
- Failure focus: action usage type lookup
- Why it fails: `action start : MissingBehaviorType05;` references an undefined behavior type.
- Typical family: `reference-error`

## 06 `06_invocation_not_behavior.sysml`
- Failure focus: invocation target typing
- Why it fails: `send Show()` invokes a non-Behavior element.
- Typical family: `invocation-expression-instantiated-type`

## 07 `07_missing_feature_in_expression.sysml`
- Failure focus: feature resolution in expression
- Why it fails: expression references `tire.height` but `height` is not declared on `Tire`.
- Typical family: `reference-error`

## 08 `08_missing_root_qualified_namespace.sysml`
- Failure focus: root-qualified namespace path
- Why it fails: `$::NoSuchRoot08::BusType` cannot resolve `NoSuchRoot08`.
- Typical family: `reference-error`

## 09 `09_redefine_missing_feature.sysml`
- Failure focus: redefinition target resolution
- Why it fails: `attribute :>> missing = 1;` attempts to redefine a non-existent inherited feature.
- Typical family: `reference-error`

## 10 `10_missing_namespace_in_type_use.sysml`
- Failure focus: qualified type namespace
- Why it fails: `MissingNS10::Wheel` references an undefined namespace.
- Typical family: `reference-error`

## Bottom line
All ten files are intentionally syntax-shaped to pass ANTLR parsing, but each introduces at least one context-sensitive violation that SysIDE correctly rejects.
