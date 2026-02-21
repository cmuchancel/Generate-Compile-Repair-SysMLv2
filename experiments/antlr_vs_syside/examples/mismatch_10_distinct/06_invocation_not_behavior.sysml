// Deterministic mismatch candidate
// Expected: ANTLR parse PASS; SysIDE compile FAIL
// Distinctness: Invocation expression targets non-behavior element.
// Expected primary SysIDE family: invocation-expression-instantiated-type
// Expected failure focus: invocation target
package Distinct06 {
  item def Picture;
  attribute def Show {
    attribute picture : Picture;
  }
  part camera {
    action takePicture {
      then send Show() via displayPort;
    }
    port displayPort;
  }
}
