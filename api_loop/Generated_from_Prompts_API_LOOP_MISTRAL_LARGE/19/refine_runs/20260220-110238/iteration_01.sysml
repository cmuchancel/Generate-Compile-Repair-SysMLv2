package PhotographyWorkflow {
    private import ScalarValues::*;

    enum def FocusStatus {
        enum Unfocused;
        enum Focused;
    }

    part def Scene {
        attribute description : String;
    }

    part def Camera {
        attribute focusStatus : FocusStatus = FocusStatus::Unfocused;
        attribute intermediateImage : Image;
        attribute finalImage : Image;

        port def focusPort;
        port def shootPort;
        port focus : focusPort;
        port shoot : shootPort;
    }

    part def Image {
        attribute data : Blob;
    }

    part def Workflow {
        part scene : Scene;
        part camera : Camera;

        attribute currentStep : String;
    }

    item def FocusCommand;
    item def ShootCommand;
    item def ImageData;

    part workflow : Workflow;

    connect workflow.camera.focus to workflow.scene;
    connect workflow.camera.shoot to workflow.camera.focus;
}