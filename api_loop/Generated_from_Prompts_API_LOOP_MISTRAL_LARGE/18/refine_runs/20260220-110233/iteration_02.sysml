package PhotoCaptureSystem {
    private import ScalarValues::*;

    enum def FocusStatus {
        enum UNFOCUSED;
        enum FOCUSING;
        enum FOCUSED;
    }

    part def Image {
    }

    part def Camera {
        attribute focusStatus : FocusStatus = FocusStatus::UNFOCUSED;
        attribute intermediateImage : Image;
        attribute finalPhoto : Image;

        port captureRequest : CaptureRequestPort;
        port focusControl : FocusControlPort;
        port imageOutput : ImageOutputPort;
    }

    part def User {
        port sceneSelection : SceneSelectionPort;
    }

    part def Scene {
    }

    port def CaptureRequestPort {
    }

    port def FocusControlPort {
    }

    port def ImageOutputPort {
    }

    port def SceneSelectionPort {
    }

    part system {
        part camera : Camera;
        part user : User;
        part scene : Scene;
    }

    connect system.user.sceneSelection to system.camera.captureRequest;
    connect system.camera.focusControl to system.camera;
    connect system.camera to system.camera.imageOutput;
}