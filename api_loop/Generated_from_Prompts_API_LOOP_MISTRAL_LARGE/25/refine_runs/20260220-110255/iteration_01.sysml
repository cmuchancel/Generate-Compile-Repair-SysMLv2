package PhotographyWorkflow {
    private import ScalarValues::*;

    enum def FocusStatus {
        enum UNFOCUSED;
        enum FOCUSING;
        enum FOCUSED;
    }

    enum def PhotoStatus {
        enum NOT_TAKEN;
        enum TAKING;
        enum TAKEN;
        enum DISPLAYED;
    }

    part def SceneSensor {
        attribute currentScene : String;
    }

    part def FocusSystem {
        attribute status : FocusStatus = FocusStatus::UNFOCUSED;
        attribute focusedImage : String;
    }

    part def Camera {
        attribute photoStatus : PhotoStatus = PhotoStatus::NOT_TAKEN;
        attribute finalPhoto : String;
    }

    part def Display {
        attribute displayedPhoto : String;
    }

    part def PhotographyController {
        port takePhoto : TakePhotoPort;
        port focusInput : FocusInputPort;
        port focusOutput : FocusOutputPort;
        port shootInput : ShootInputPort;
        port shootOutput : ShootOutputPort;
        port displayInput : DisplayInputPort;
    }

    port def TakePhotoPort;
    port def FocusInputPort;
    port def FocusOutputPort;
    port def ShootInputPort;
    port def ShootOutputPort;
    port def DisplayInputPort;

    part photographySystem {
        part sceneSensor : SceneSensor;
        part focusSystem : FocusSystem;
        part camera : Camera;
        part display : Display;
        part controller : PhotographyController;

        connect controller.takePhoto to controller.focusInput;
        connect controller.focusOutput to controller.shootInput;
        connect controller.shootOutput to controller.displayInput;
    }
}