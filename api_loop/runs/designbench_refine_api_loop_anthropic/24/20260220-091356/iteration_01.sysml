package CameraWorkflow {
    private import ScalarValues::*;

    enum def FocusStatus {
        enum WellFocused;
        enum PoorFocus;
    }

    enum def ShootingStatus {
        enum Ready;
        enum NotReady;
    }

    item def Scene;
    item def FocusedImage;
    item def Photograph;

    port def SceneInputPort {
        attribute sceneData : String;
    }

    port def FocusedImagePort {
        attribute focusQuality : FocusStatus;
    }

    port def PhotoOutputPort {
        attribute photoReady : ShootingStatus;
    }

    part def SceneSelector {
        port sceneOut : SceneInputPort;
        attribute selectedScene : String;
    }

    part def FocusEngine {
        port sceneIn : SceneInputPort;
        port focusOut : FocusedImagePort;
        attribute focusResult : FocusStatus;
        attribute focusAchieved : Boolean;
    }

    part def FocusEvaluator {
        port focusIn : FocusedImagePort;
        port evaluationOut : FocusedImagePort;
        attribute evaluationResult : FocusStatus;
        attribute isWellFocused : Boolean;
    }

    part def ShootingEngine {
        port focusCheckIn : FocusedImagePort;
        port photoOut : PhotoOutputPort;
        attribute shootingEnabled : Boolean;
        attribute shootingStatus : ShootingStatus;
    }

    part def PhotoOutput {
        port photoIn : PhotoOutputPort;
        attribute finalPhoto : String;
        attribute photoDelivered : Boolean;
    }

    part def CameraSystem {
        part selector : SceneSelector;
        part focusEngine : FocusEngine;
        part evaluator : FocusEvaluator;
        part shooter : ShootingEngine;
        part output : PhotoOutput;

        connect selector.sceneOut to focusEngine.sceneIn;
        connect focusEngine.focusOut to evaluator.focusIn;
        connect evaluator.evaluationOut to shooter.focusCheckIn;
        connect shooter.photoOut to output.photoIn;
    }
}