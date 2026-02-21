// Deterministic mismatch candidate
// Expected: ANTLR parse PASS; SysIDE compile FAIL
// Distinctness: Specialization base classifier does not exist.
// Expected primary SysIDE family: reference-error
// Expected failure focus: specialization target
package Distinct04 {
  part def Motor :> MissingBaseMotor04 {
  }
}
