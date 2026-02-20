package PhotographyWorkflow {
    private import ScalarValues::*;

    enum def WorkflowStep {
        enum Focusing;
        enum Shooting;
    }

    enum def ImageState {
        enum Unfocused;
        enum Focused;
        enum Final;
    }

    item def Scene;
    item def IntermediateImage;
    item def FinalPhoto;

    port def ScenePort;
    port def ImagePort;
    port def PhotoPort;

    part def User {
        port sceneOut : ScenePort;
    }

    part def FocusingUnit {
        port sceneIn : ScenePort;
        port imageOut : ImagePort;
        attribute currentStep : WorkflowStep = WorkflowStep::Focusing;
        attribute outputState : ImageState = ImageState::Focused;
    }

    part def ShootingUnit {
        port imageIn : ImagePort;
        port photoOut : PhotoPort;
        attribute currentStep : WorkflowStep = WorkflowStep::Shooting;
        attribute outputState : ImageState = ImageState::Final;
    }

    part def Camera {
        port sceneIn : ScenePort;
        port photoOut : PhotoPort;

        part user : User;
        part focusingUnit : FocusingUnit;
        part shootingUnit : ShootingUnit;

        connect user.sceneOut to focusingUnit.sceneIn;
        connect focusingUnit.imageOut to shootingUnit.imageIn;
        connect shootingUnit.photoOut to photoOut;
    }
}