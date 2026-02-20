package CameraSystem {
    private import ScalarValues::*;

    item def Image;
    item def Picture;

    port def ViewPort;
    port def DisplayPort;

    part def Scene;

    part def Viewfinder {
        port viewPort : ViewPort;
    }

    part def FocusUnit {
        attribute focused : Boolean = false;
        item focusedImage : Image;
    }

    part def CaptureUnit {
        item capturedPicture : Picture;
    }

    part def DisplayUnit {
        port displayPort : DisplayPort;
    }

    part def Camera {
        part viewfinder : Viewfinder;
        part focusUnit : FocusUnit;
        part captureUnit : CaptureUnit;
        part displayUnit : DisplayUnit;

        port viewPort : ViewPort;
        port displayPort : DisplayPort;

        connect viewPort to viewfinder.viewPort;
        connect displayUnit.displayPort to displayPort;
    }

    part def User {
        port userViewPort : ViewPort;
        port userDisplayPort : DisplayPort;
    }

    part def CameraSystem {
        part camera : Camera;
        part user : User;

        connect user.userViewPort to camera.viewPort;
        connect camera.displayPort to user.userDisplayPort;
    }
}