package AutomatedPhotographySystem {
    private import ScalarValues::*;

    item def ShootingScene;
    
    item def ProcessedImage;
    
    item def Photograph;
    
    part def FocusProcessor {
        port input sceneInput : ShootingScene;
        port output imageOutput : ProcessedImage;
    }
    
    part def Shooter {
        port input imageInput : ProcessedImage;
        port output photoOutput : Photograph;
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