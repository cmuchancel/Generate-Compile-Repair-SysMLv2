package CameraSystem {
    private import ScalarValues::*;

    enum def FocusStatus {
        enum UNFOCUSED;
        enum FOCUSING;
        enum FOCUSED;
    }

    part def Camera {
        attribute focusStatus : FocusStatus = FocusStatus::UNFOCUSED;
        attribute isCapturing : Boolean = false;
        attribute isDisplaying : Boolean = false;

        port def SceneInputPort;
        port def ImageOutputPort;
        port def DisplayControlPort;

        part sceneInput : SceneInputPort;
        part imageOutput : ImageOutputPort;
        part displayControl : DisplayControlPort;
    }

    part def Display {
        port def ImageInputPort;
        port imageInput : ImageInputPort;
    }

    part def Scene {
    }

    item def Image {
    }

    part camera : Camera;
    part display : Display;
    part scene : Scene;
    part photo : Image;

    connect camera.sceneInput to scene;
    connect camera.imageOutput to photo;
    connect camera.displayControl to display.imageInput;
}