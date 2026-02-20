package PhotographyWorkflow {
    private import ScalarValues::*;

    enum def FocusStatus {
        enum UNFOCUSED;
        enum FOCUSING;
        enum FOCUSED;
    }

    enum def ShootingStatus {
        enum READY;
        enum SHOOTING;
        enum COMPLETED;
    }

    part def Scene {
        attribute selected : Boolean = false;
    }

    part def Image {
        attribute resolution : String;
    }

    part def Photograph {
        attribute quality : String;
    }

    part def Camera {
        attribute focusStatus : FocusStatus = FocusStatus::UNFOCUSED;
        attribute shootingStatus : ShootingStatus = ShootingStatus::READY;

        port focusControl : FocusControlPort;
        port shootingControl : ShootingControlPort;
        port imageOutput : ImageOutputPort;
    }

    part def User {
        port sceneSelection : SceneSelectionPort;
    }

    port def FocusControlPort;
    port def ShootingControlPort;
    port def ImageOutputPort;
    port def SceneSelectionPort;

    part workflowSystem {
        part user : User;
        part camera : Camera;
        part scene : Scene;
        part image : Image;
        part photograph : Photograph;

        connect user.sceneSelection to scene;
        connect camera.focusControl to scene;
        connect camera.shootingControl to image;
        connect camera.imageOutput to photograph;
    }
}