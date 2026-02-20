package PhotoSystem {
    private import ScalarValues::*;

    item def Image {
        attribute id : String;
        attribute isFocused : Boolean;
    }

    port def ImagePort {
        attribute image : Image;
    }

    part def FocusModule {
        port input : ImagePort;
        port output : ImagePort;
        attribute intermediateImage : Image;
    }

    part def CaptureModule {
        port input : ImagePort;
        port output : ImagePort;
        attribute finalPhoto : Image;
    }

    part def User {
        port sceneSelection : ImagePort;
        port photoDelivery : ImagePort;
    }

    part def CameraSystem {
        part focusModule : FocusModule;
        part captureModule : CaptureModule;
        port systemInput : ImagePort;
        port systemOutput : ImagePort;

        connect systemInput to focusModule.input;
        connect focusModule.output to captureModule.input;
        connect captureModule.output to systemOutput;
    }

    part user : User;
    part cameraSystem : CameraSystem;

    connect user.sceneSelection to cameraSystem.systemInput;
    connect cameraSystem.systemOutput to user.photoDelivery;
}