package AutomatedPhotographySystem {
    private import ScalarValues::*;

    item def ShootingScene;

    item def ProcessedImage;

    item def Photograph;

    part def FocusProcessor {
        port sceneInput : ShootingScene;
        port imageOutput : ProcessedImage;
    }

    part def Shooter {
        port imageInput : ProcessedImage;
        port photoOutput : Photograph;
    }

    part def AutomatedPhotographySystem {
        part focusProcessor : FocusProcessor;
        part shooter : Shooter;

        port systemInput : ShootingScene;
        port systemOutput : Photograph;

        connect systemInput to focusProcessor.sceneInput;
        connect focusProcessor.imageOutput to shooter.imageInput;
        connect shooter.photoOutput to systemOutput;
    }
}