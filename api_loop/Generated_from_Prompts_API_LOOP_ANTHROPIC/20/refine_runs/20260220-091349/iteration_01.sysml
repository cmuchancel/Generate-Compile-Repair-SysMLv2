package PhotographyWorkflow {
    private import ScalarValues::*;

    enum def WorkflowStatus {
        enum PENDING;
        enum FOCUSING;
        enum FOCUSED;
        enum SHOOTING;
        enum COMPLETE;
    }

    item def Scene;
    item def Image;
    item def Photograph;

    port def SceneInputPort {
        attribute sceneReady : Boolean;
    }

    port def ImageOutputPort {
        attribute imageReady : Boolean;
    }

    port def PhotographOutputPort {
        attribute photographReady : Boolean;
    }

    part def UserInterface {
        attribute selectedScene : String;
        port sceneOut : SceneInputPort;
    }

    part def FocusingUnit {
        attribute status : WorkflowStatus;
        attribute focusAchieved : Boolean;
        port sceneIn : SceneInputPort;
        port imageOut : ImageOutputPort;
    }

    part def ShootingUnit {
        attribute status : WorkflowStatus;
        attribute photographProduced : Boolean;
        port imageIn : ImageOutputPort;
        port photographOut : PhotographOutputPort;
    }

    part def PhotographySystem {
        attribute workflowStatus : WorkflowStatus;
        attribute autoWorkflowEnabled : Boolean = true;

        part ui : UserInterface;
        part focusingUnit : FocusingUnit;
        part shootingUnit : ShootingUnit;

        connect ui.sceneOut to focusingUnit.sceneIn;
        connect focusingUnit.imageOut to shootingUnit.imageIn;
    }
}