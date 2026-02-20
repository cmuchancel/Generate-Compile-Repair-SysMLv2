package PhotographyWorkflow {
    private import ScalarValues::*;

    enum def FocusStatus {
        enum Clear;
        enum Unclear;
    }

    enum def ProcessStatus {
        enum Pending;
        enum InProgress;
        enum Completed;
    }

    item def Scene;
    item def Image;

    port def SceneInputPort {
        attribute sceneReady : Boolean;
    }

    port def ImageOutputPort {
        attribute imageReady : Boolean;
    }

    port def FocusPort {
        attribute focusStatus : FocusStatus;
    }

    part def Camera {
        port sceneIn : SceneInputPort;
        port imageOut : ImageOutputPort;
        port focusPort : FocusPort;
        attribute cameraStatus : ProcessStatus;
    }

    part def FocusSystem {
        port sceneIn : SceneInputPort;
        port focusOut : FocusPort;
        attribute focusResult : FocusStatus;
        attribute focusAchieved : Boolean;
    }

    part def ShootingSystem {
        port focusIn : FocusPort;
        port imageOut : ImageOutputPort;
        attribute shootingStatus : ProcessStatus;
        attribute imageGenerated : Boolean;
    }

    part def UserInterface {
        port sceneOut : SceneInputPort;
        attribute selectedScene : Boolean;
        attribute oneShotMode : Boolean;
    }

    part def PhotographySystem {
        part ui : UserInterface;
        part camera : Camera;
        part focusSystem : FocusSystem;
        part shootingSystem : ShootingSystem;

        attribute workflowStatus : ProcessStatus;
        attribute finalImageReady : Boolean;

        connect ui.sceneOut to camera.sceneIn;
        connect camera.sceneIn to focusSystem.sceneIn;
        connect focusSystem.focusOut to camera.focusPort;
        connect camera.focusPort to shootingSystem.focusIn;
        connect shootingSystem.imageOut to camera.imageOut;
    }
}