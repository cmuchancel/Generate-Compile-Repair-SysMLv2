package PhotographyWorkflow {
    private import ScalarValues::*;

    item def Scene {
        attribute description : String;
    }

    item def Image {
        attribute data : String;
        attribute focused : Boolean = false;
    }

    port def FocusPort {
        attribute target : Scene;
        attribute result : Image;
    }

    port def ShootPort {
        attribute input : Image;
        attribute output : Image;
    }

    part def Focusing {
        port focusIO : FocusPort;
        attribute selectedScene : Scene;
        attribute intermediateImage : Image;
    }

    part def Shooting {
        port shootIO : ShootPort;
        attribute finalPhoto : Image;
    }

    part def PhotographySystem {
        part focusUnit : Focusing;
        part shootUnit : Shooting;
        port systemInput : FocusPort;
        port systemOutput : ShootPort;

        connect systemInput to focusUnit.focusIO;
        connect focusUnit.focusIO to shootUnit.shootIO;
        connect shootUnit.shootIO to systemOutput;
    }
}