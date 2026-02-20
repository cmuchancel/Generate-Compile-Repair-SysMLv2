package PhotographyWorkflow {
    private import ScalarValues::*;

    enum def FocusStatus {
        enum Clear;
        enum NotClear;
    }

    item def Scene {}
    item def Photo {}

    port def TriggerPort {}
    port def FocusCheckPort {}
    port def CapturePort {}

    part def Camera {
        port triggerInput : TriggerPort;
        port focusCheck : FocusCheckPort;
        port capture : CapturePort;
        attribute focusStatus : FocusStatus;
        attribute capturedPhoto : Photo;
    }

    part def User {
        port triggerOutput : TriggerPort;
        attribute selectedScene : Scene;
    }

    part def Environment {
        port focusSubject : FocusCheckPort;
        port captureSubject : CapturePort;
        attribute currentScene : Scene;
    }

    part camera : Camera;
    part user : User;
    part environment : Environment;

    connect user.triggerOutput to camera.triggerInput;
    connect camera.focusCheck to environment.focusSubject;
    connect camera.capture to environment.captureSubject;
}