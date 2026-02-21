// Deterministic mismatch candidate
// Expected: ANTLR parse PASS; SysIDE compile FAIL
// Distinctness: Missing namespace in import statement.
// Expected primary SysIDE family: reference-error
// Expected failure focus: import line
package Distinct01 {
  private import MissingPkg01::*;
  part def A {}
}
