package CameraSystem {
    private import ScalarValues::*;

    item def Scene {
        attribute description : String;
    }

    item def Image {
        attribute data : String;
    }

    item def Photo {
        attribute data : String;
    }

    part def FocusingUnit {
        port input : Scene;
        port output : Image;
    }

    part def CapturingUnit {
        port input : Image;
        port output : Photo;
    }

    part def DisplayUnit {
        port input : Photo;
    }

    part def Camera {
        part focusingUnit : FocusingUnit;
        part capturingUnit : CapturingUnit;
        part displayUnit : DisplayUnit;
        port sceneInput : Scene;

        connect sceneInput to focusingUnit.input;
        connect focusingUnit.output to capturingUnit.input;
        connect capturingUnit.output to displayUnit.input;
    }

    part def CameraSystem {
        part camera : Camera;
        port userSceneInput : Scene;

        connect userSceneInput to camera.sceneInput;
    }
}