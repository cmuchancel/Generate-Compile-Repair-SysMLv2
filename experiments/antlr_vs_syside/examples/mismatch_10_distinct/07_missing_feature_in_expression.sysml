// Deterministic mismatch candidate
// Expected: ANTLR parse PASS; SysIDE compile FAIL
// Distinctness: Expression references undefined feature on a known part.
// Expected primary SysIDE family: reference-error
// Expected failure focus: feature in expression
package Distinct07 {
  attribute def LengthValue;
  part def Tire {
    attribute width: LengthValue;
  }
  part def Wheel {
    attribute hubDiameter: LengthValue;
    attribute outerDiameter: LengthValue = (hubDiameter + 2 * tire.height);
    part tire: Tire[1];
  }
}
