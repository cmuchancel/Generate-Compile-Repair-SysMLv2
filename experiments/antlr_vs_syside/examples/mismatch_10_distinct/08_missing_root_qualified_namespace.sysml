// Deterministic mismatch candidate
// Expected: ANTLR parse PASS; SysIDE compile FAIL
// Distinctness: Global/root qualified namespace does not exist.
// Expected primary SysIDE family: reference-error
// Expected failure focus: root-qualified namespace
package Distinct08 {
  part def Controller {
    port comms: $::NoSuchRoot08::BusType;
  }
}
