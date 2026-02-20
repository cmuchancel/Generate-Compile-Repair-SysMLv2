package CameraSystem {
    private import ScalarValues::*;

    item def Scene {
        attribute description : string;
    }

    item def Image {
        attribute data : string;
    }

    item def Photo {
        attribute data : string;
    }

    part def FocusingUnit {
        port in input : Scene;
        port out output : Image;
    }

    part def CapturingUnit {
        port in input : Image;
        port out output : Photo;
    }

    part def DisplayUnit {
        port in input : Photo;
    }

    part def Camera {
        part focusingUnit : FocusingUnit;
        part capturingUnit : CapturingUnit;
        part displayUnit : DisplayUnit;

        port in sceneInput : Scene;

        connect sceneInput to focusingUnit.input;
        connect focusingUnit.output to capturingUnit.input;
        connect capturingUnit.output to displayUnit.input;
    }

    part def CameraSystem {
        part camera : Camera;
        port in userSceneInput : Scene;

        connect userSceneInput to camera.sceneInput;
    }
}