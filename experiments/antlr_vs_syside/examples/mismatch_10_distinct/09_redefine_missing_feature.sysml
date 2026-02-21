// Deterministic mismatch candidate
// Expected: ANTLR parse PASS; SysIDE compile FAIL
// Distinctness: Redefinition target is missing in specialized type.
// Expected primary SysIDE family: reference-error
// Expected failure focus: redefinition target
package Distinct09 {
  attribute def NumberLike;
  part def Base {
    attribute a: NumberLike;
  }
  part def Derived :> Base {
    attribute :>> missing = 1;
  }
}
