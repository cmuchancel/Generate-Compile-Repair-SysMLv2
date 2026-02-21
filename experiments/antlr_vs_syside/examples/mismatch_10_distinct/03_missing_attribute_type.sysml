// Deterministic mismatch candidate
// Expected: ANTLR parse PASS; SysIDE compile FAIL
// Distinctness: Missing type on an attribute declaration.
// Expected primary SysIDE family: reference-error
// Expected failure focus: attribute type reference
package Distinct03 {
  part def Battery {
    attribute nominalVoltage: MissingVoltageType03;
  }
}
