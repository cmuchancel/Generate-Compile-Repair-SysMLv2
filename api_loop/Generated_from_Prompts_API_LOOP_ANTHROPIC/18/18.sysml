package CameraSystem {
    private import ScalarValues::*;

    enum def FocusStatus {
        enum Unfocused;
        enum Focused;
    }

    enum def CaptureStatus {
        enum Idle;
        enum Capturing;
        enum Done;
    }

    item def Scene;
    item def IntermediateImage;
    item def Photo;

    port def SceneInputPort {
        attribute sceneData : String;
    }

    port def ImageOutputPort {
        attribute imageData : String;
    }

    port def PhotoOutputPort {
        attribute photoData : String;
    }

    part def User {
        port sceneOut : SceneInputPort;
        port photoIn : PhotoOutputPort;
    }

    part def FocusModule {
        attribute focusStatus : FocusStatus;
        port sceneIn : SceneInputPort;
        port imageOut : ImageOutputPort;
    }

    part def CaptureModule {
        attribute captureStatus : CaptureStatus;
        port imageIn : ImageOutputPort;
        port photoOut : PhotoOutputPort;
    }

    part def CameraSystem {
        attribute systemName : String = "CameraSystem";

        port userSceneIn : SceneInputPort;
        port userPhotoOut : PhotoOutputPort;

        part focusModule : FocusModule;
        part captureModule : CaptureModule;

        connect userSceneIn to focusModule.sceneIn;
        connect focusModule.imageOut to captureModule.imageIn;
        connect captureModule.photoOut to userPhotoOut;
    }

    part user : User;
    part camera : CameraSystem;

    connect user.sceneOut to camera.userSceneIn;
    connect camera.userPhotoOut to user.photoIn;
}