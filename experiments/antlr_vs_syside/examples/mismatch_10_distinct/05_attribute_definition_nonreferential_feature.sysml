// Deterministic mismatch candidate
// Expected: ANTLR parse PASS; SysIDE compile FAIL
// Distinctness: Missing type on action usage declaration.
// Expected primary SysIDE family: reference-error
// Expected failure focus: action type reference
package Distinct05 {
  part def OperatorConsole {
    action start : MissingBehaviorType05;
  }
}
