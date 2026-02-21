// Deterministic mismatch candidate
// Expected: ANTLR parse PASS; SysIDE compile FAIL
// Distinctness: Namespace in qualified type use does not exist.
// Expected primary SysIDE family: reference-error
// Expected failure focus: qualified type namespace
package Distinct10 {
  part def Vehicle {
    part wheel: MissingNS10::Wheel;
  }
}
