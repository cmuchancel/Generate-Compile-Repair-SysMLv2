package CameraSystem {
    private import ScalarValues::*;

    enum def FocusStatus {
        enum UNFOCUSED;
        enum FOCUSING;
        enum FOCUSED;
    }

    enum def ImagingStatus {
        enum IDLE;
        enum CAPTURING;
        enum COMPLETED;
    }

    part def Camera {
        attribute focusStatus : FocusStatus = FocusStatus::UNFOCUSED;
        attribute imagingStatus : ImagingStatus = ImagingStatus::IDLE;
        attribute photoCount : Integer = 0;

        part autofocus : AutofocusSubsystem;
        part imaging : ImagingSubsystem;
        part storage : StorageSubsystem;
    }

    part def AutofocusSubsystem {
        attribute currentFocus : Real = 0.0;
        attribute targetFocus : Real = 0.0;
    }

    part def ImagingSubsystem {
        attribute resolution : String = "1920x1080";
    }

    part def StorageSubsystem {
        attribute capacity : Integer = 100;
    }

    part def User {
        port sceneSelection : SceneSelectionPort;
    }

    port def SceneSelectionPort;

    part camera : Camera;
    part user : User;

    connect user.sceneSelection to camera.autofocus;
}