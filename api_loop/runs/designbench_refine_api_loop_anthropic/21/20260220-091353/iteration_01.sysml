package AutomatedPhotography {
    private import ScalarValues::*;

    enum def ShootingStatus {
        enum IDLE;
        enum FOCUSING;
        enum FOCUSED;
        enum SHOOTING;
        enum COMPLETE;
    }

    item def Scene;
    item def FocusedImage;
    item def Photograph;

    port def SceneInputPort {
        attribute sceneDescription : String;
    }

    port def FocusedImagePort {
        attribute imageReady : Boolean;
    }

    port def PhotographPort {
        attribute photographReady : Boolean;
    }

    part def FocusProcessor {
        attribute status : ShootingStatus;
        attribute focusQuality : Real;
        port sceneIn : SceneInputPort;
        port imageOut : FocusedImagePort;
    }

    part def ShootingProcessor {
        attribute status : ShootingStatus;
        attribute shutterSpeed : Real;
        attribute aperture : Real;
        port imageIn : FocusedImagePort;
        port photographOut : PhotographPort;
    }

    part def UserInterface {
        attribute shootingScene : String;
        port sceneOut : SceneInputPort;
        port finalPhotoIn : PhotographPort;
    }

    part def AutomatedPhotographySystem {
        attribute systemName : String = "AutomatedPhotographySystem";
        attribute systemStatus : ShootingStatus;

        part ui : UserInterface;
        part focusProcessor : FocusProcessor;
        part shootingProcessor : ShootingProcessor;

        port systemSceneIn : SceneInputPort;
        port systemPhotoOut : PhotographPort;

        connect ui.sceneOut to focusProcessor.sceneIn;
        connect focusProcessor.imageOut to shootingProcessor.imageIn;
        connect shootingProcessor.photographOut to ui.finalPhotoIn;
    }
}