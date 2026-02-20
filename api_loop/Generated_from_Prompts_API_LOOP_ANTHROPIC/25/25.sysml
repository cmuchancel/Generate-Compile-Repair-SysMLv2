package PhotographyWorkflow {
    private import ScalarValues::*;

    enum def WorkflowStatus {
        enum IDLE;
        enum FOCUSING;
        enum SHOOTING;
        enum DISPLAYING;
        enum COMPLETE;
    }

    item def Scene;
    item def Image;
    item def Photo;

    port def ScenePort {
        attribute sceneData : String;
    }

    port def ImagePort {
        attribute imageData : String;
    }

    port def PhotoPort {
        attribute photoData : String;
    }

    part def FocusUnit {
        port sceneIn : ScenePort;
        port imageOut : ImagePort;
        attribute status : WorkflowStatus;
        attribute focusComplete : Boolean;
    }

    part def CameraUnit {
        port imageIn : ImagePort;
        port photoOut : PhotoPort;
        attribute status : WorkflowStatus;
        attribute shootComplete : Boolean;
    }

    part def DisplayUnit {
        port photoIn : PhotoPort;
        attribute status : WorkflowStatus;
        attribute displayComplete : Boolean;
    }

    part def UserInterface {
        port sceneOut : ScenePort;
        attribute triggerTakePhoto : Boolean;
        attribute currentScene : String;
    }

    part def PhotographySystem {
        attribute systemStatus : WorkflowStatus;
        attribute workflowActive : Boolean;

        part userInterface : UserInterface;
        part focusUnit : FocusUnit;
        part cameraUnit : CameraUnit;
        part displayUnit : DisplayUnit;

        connect userInterface.sceneOut to focusUnit.sceneIn;
        connect focusUnit.imageOut to cameraUnit.imageIn;
        connect cameraUnit.photoOut to displayUnit.photoIn;
    }
}