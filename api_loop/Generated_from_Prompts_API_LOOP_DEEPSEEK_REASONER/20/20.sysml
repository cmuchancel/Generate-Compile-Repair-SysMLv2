package PhotographyWorkflow {
    private import ScalarValues::*;

    item def Scene;
    item def Image;
    item def Photograph;

    port def SceneSelectionPort {
        attribute selectedScene : Scene;
    }

    port def ImagePort {
        attribute image : Image;
    }

    port def PhotographPort {
        attribute photograph : Photograph;
    }

    part def User {
        port sceneSelection : SceneSelectionPort;
    }

    part def FocusingProcess {
        port input : SceneSelectionPort;
        port output : ImagePort;
        attribute focusImage : Image;
    }

    part def ShootingProcess {
        port input : ImagePort;
        port output : PhotographPort;
        attribute finalPhotograph : Photograph;
    }

    part def PhotographySystem {
        part user : User;
        part focuser : FocusingProcess;
        part shooter : ShootingProcess;

        connect user.sceneSelection to focuser.input;
        connect focuser.output to shooter.input;
    }
}