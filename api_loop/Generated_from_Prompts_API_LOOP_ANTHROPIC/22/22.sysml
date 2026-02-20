package PhotographyWorkflow {
    private import ScalarValues::*;

    item def Scene;
    item def Image;
    item def Picture;

    port def ScenePort {
        attribute sceneData : String;
    }

    port def ImagePort {
        attribute imageData : String;
    }

    port def PicturePort {
        attribute pictureData : String;
    }

    part def Focus {
        port sceneIn : ScenePort;
        port imageOut : ImagePort;
    }

    part def Shoot {
        port imageIn : ImagePort;
        port pictureOut : PicturePort;
    }

    part def PhotographyWorkflowSystem {
        port sceneInput : ScenePort;
        port pictureOutput : PicturePort;

        part focus : Focus;
        part shoot : Shoot;

        connect sceneInput to focus::sceneIn;
        connect focus::imageOut to shoot::imageIn;
        connect shoot::pictureOut to pictureOutput;
    }
}