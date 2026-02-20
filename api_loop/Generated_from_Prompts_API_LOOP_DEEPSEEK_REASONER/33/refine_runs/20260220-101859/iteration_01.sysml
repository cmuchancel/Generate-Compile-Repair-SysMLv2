package CameraSystem {
    private import ScalarValues::*;

    item def Image {
        attribute data : String;
    }

    item def Picture {
        attribute data : String;
    }

    port def ViewPort {
    }

    port def DisplayPort {
    }

    port def FocusPort {
    }

    port def CapturePort {
    }

    part def Camera {
        port viewPort : ViewPort;
        port displayPort : DisplayPort;
        part focus : Focus;
        part capture : Capture;
        connect viewPort to focus.viewPort;
        connect focus.focusPort to capture.focusPort;
        connect capture.displayPort to displayPort;
    }

    part def Focus {
        port viewPort : ViewPort;
        port focusPort : FocusPort;
        attribute currentImage : Image;
    }

    part def Capture {
        port focusPort : FocusPort;
        port displayPort : DisplayPort;
        attribute currentPicture : Picture;
    }
}