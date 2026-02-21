// Deterministic mismatch candidate
// Expected: ANTLR parse PASS; SysIDE compile FAIL
// Distinctness: Missing type on a port declaration.
// Expected primary SysIDE family: reference-error
// Expected failure focus: port type reference
package Distinct02 {
  part def Controller {
    port controlPort: MissingPortType02;
  }
}
