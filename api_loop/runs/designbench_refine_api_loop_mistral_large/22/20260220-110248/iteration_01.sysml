package PhotographyWorkflow {
    private import ScalarValues::*;

    enum def FocusStatus {
        enum Unfocused;
        enum Focused;
    }

    part def Scene;
    part def Image;
    part def Picture;

    part def Camera {
        attribute focusStatus : FocusStatus = FocusStatus::Unfocused;
        port focusPort : FocusPort;
        port shootPort : ShootPort;
    }

    part def PhotographySystem {
        part camera : Camera;
        part inputScene : Scene;
        part outputPicture : Picture;
    }

    port def FocusPort;
    port def ShootPort;

    item def FocusCommand;
    item def ShootCommand;

    part photographyWorkflow : PhotographySystem;

    connect photographyWorkflow.camera.focusPort to photographyWorkflow.inputScene;
    connect photographyWorkflow.camera.shootPort to photographyWorkflow.outputPicture;
}