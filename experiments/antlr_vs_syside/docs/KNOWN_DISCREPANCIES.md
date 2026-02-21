# Known Discrepancies

This file tracks known cases where third-party HAMR ANTLR parsing and SysIDE compilation disagree.

## Case: `Generated_from_Prompts_API_LOOP_MISTRAL_LARGE/60/60.sysml`

- File: `api_loop/Generated_from_Prompts_API_LOOP_MISTRAL_LARGE/60/60.sysml`
- Observed in full generated-file audit:
  - ANTLR (HAMR) parse: **FAIL**
  - SysIDE compile: **PASS**

### Reported ANTLR diagnostics

```text
line 31:13 no viable alternative at input 'port compute'
line 39:38 mismatched input 'compute' expecting {'$', RULE_ID, RULE_UNRESTRICTED_NAME}
line 40:42 mismatched input 'compute' expecting {'$', RULE_ID, RULE_UNRESTRICTED_NAME}
```

### Interpretation

The identifier `compute` is accepted by SysIDE in this context but rejected by the HAMR parser tokenization/grammar path as a non-identifier token. This is a parser/tooling divergence, not a SysIDE compilation failure.

### Practical handling

- Do not treat this file as a semantic SysIDE failure.
- Track it as a parser-compatibility edge case when reporting ANTLR-vs-SysIDE comparisons.
