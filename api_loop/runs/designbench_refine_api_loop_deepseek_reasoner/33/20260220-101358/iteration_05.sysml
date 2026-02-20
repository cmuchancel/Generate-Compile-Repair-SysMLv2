package CameraSystem {
    private import ScalarValues::*;

    enum def FocusStatus {
        enum FOCUSED;
        enum NOT_FOCUSED;
    }

    item def Scene;
    item def Image;
    item def Picture;

    port def viewPort {
        attribute selectedScene : Scene;
    }

    port def imagePort {
        attribute image : Image;
    }

    port def picturePort {
        attribute picture : Picture;
    }

    port def displayPort {
        attribute displayedPicture : Picture;
    }

    part def FocusModule {
        port input : viewPort;
        port output : imagePort;
        attribute status : FocusStatus = FocusStatus::NOT_FOCUSED;
    }

    part def CaptureModule {
        port input : imagePort;
        port output : picturePort;
    }

    part def DisplayModule {
        port input : picturePort;
        port output : displayPort;
    }

    part def Camera {
        port view : viewPort;
        port displayOut : displayPort;
        part focus : FocusModule;
        part capture : CaptureModule;
        part display : DisplayModule;

        connect view to focus.input;
        connect focus.output to capture.input;
        connect capture.output to display.input;
        connect display.output to displayOut;
    }
}