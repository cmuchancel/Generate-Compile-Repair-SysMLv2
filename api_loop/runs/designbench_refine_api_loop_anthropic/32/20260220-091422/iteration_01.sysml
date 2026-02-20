package CameraSystem {
    private import ScalarValues::*;

    enum def FocusState {
        enum Unfocused;
        enum Focusing;
        enum Focused;
    }

    enum def CaptureState {
        enum Idle;
        enum Capturing;
        enum Captured;
    }

    enum def DisplayState {
        enum Off;
        enum Displaying;
    }

    item def Scene;
    item def Image;
    item def Photo;

    port def SceneInputPort {
        attribute sceneReady : Boolean;
    }

    port def ImagePort {
        attribute imageAvailable : Boolean;
    }

    port def PhotoPort {
        attribute photoReady : Boolean;
    }

    port def DisplayPort {
        attribute displayActive : Boolean;
    }

    part def User {
        port sceneOut : SceneInputPort;
    }

    part def FocusUnit {
        attribute focusState : FocusState;
        port sceneIn : SceneInputPort;
        port imageOut : ImagePort;
    }

    part def CaptureUnit {
        attribute captureState : CaptureState;
        port imageIn : ImagePort;
        port photoOut : PhotoPort;
    }

    part def DisplayUnit {
        attribute displayState : DisplayState;
        port photoIn : PhotoPort;
        port displayOut : DisplayPort;
    }

    part def Screen {
        attribute active : Boolean;
        port displayIn : DisplayPort;
    }

    part def Camera {
        attribute cameraId : String;

        part user : User;
        part focusUnit : FocusUnit;
        part captureUnit : CaptureUnit;
        part displayUnit : DisplayUnit;
        part screen : Screen;

        connect user.sceneOut to focusUnit.sceneIn;
        connect focusUnit.imageOut to captureUnit.imageIn;
        connect captureUnit.photoOut to displayUnit.photoIn;
        connect displayUnit.displayOut to screen.displayIn;
    }
}