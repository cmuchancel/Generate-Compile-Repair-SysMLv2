package AutomatedPhotographySystem {
    private import ScalarValues::*;

    item def ShootingScene;

    item def ProcessedImage;

    item def Photograph;

    part def FocusProcessor {
        port sceneInput : in ShootingScene;
        port imageOutput : out ProcessedImage;
    }

    part def Shooter {
        port imageInput : in ProcessedImage;
        port photoOutput : out Photograph;
    }

    part def AutomatedPhotographySystem {
        part focusProcessor : FocusProcessor;
        part shooter : Shooter;

        port systemInput : in ShootingScene;
        port systemOutput : out Photograph;

        connect systemInput to focusProcessor.sceneInput;
        connect focusProcessor.imageOutput to shooter.imageInput;
        connect shooter.photoOutput to systemOutput;
    }
}