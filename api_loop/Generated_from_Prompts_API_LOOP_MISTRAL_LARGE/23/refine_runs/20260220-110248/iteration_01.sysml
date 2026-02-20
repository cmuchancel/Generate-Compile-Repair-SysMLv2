package PhotographyWorkflow {
    private import ScalarValues::*;

    enum def FocusStatus {
        enum CLEAR
        enum UNCLEAR
    }

    part def Camera {
        attribute focusStatus : FocusStatus;
        attribute isShooting : Boolean = false;

        port def FocusPort;
        port def CapturePort;

        port focus : FocusPort;
        port capture : CapturePort;
    }

    part def Scene {
        attribute isSelected : Boolean = false;
    }

    part def Image {
        attribute isGenerated : Boolean = false;
    }

    part def PhotographySystem {
        part camera : Camera;
        part scene : Scene;
        part image : Image;

        attribute currentFocusStatus : FocusStatus = FocusStatus::UNCLEAR;

        connect camera.focus to scene;
        connect camera.capture to image;
    }
}