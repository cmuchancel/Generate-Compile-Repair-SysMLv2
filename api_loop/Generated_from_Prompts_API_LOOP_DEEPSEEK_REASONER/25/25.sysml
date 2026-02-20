package PhotographyWorkflow {
    private import ScalarValues::*;
    
    item def Scene;
    item def FocusedImage;
    item def Photo;
    
    part def Camera {
        port sceneInput : Scene;
        port focusOutput : FocusedImage;
        port photoOutput : Photo;
    }
    
    part def Display {
        port photoInput : Photo;
    }
    
    part def FocusModule {
        port sceneInput : Scene;
        port imageOutput : FocusedImage;
    }
    
    part def ShootModule {
        port imageInput : FocusedImage;
        port photoOutput : Photo;
    }
    
    part def PhotographySystem {
        part camera : Camera;
        part focusModule : FocusModule;
        part shootModule : ShootModule;
        part display : Display;
        
        connect camera.sceneInput to focusModule.sceneInput;
        connect focusModule.imageOutput to shootModule.imageInput;
        connect shootModule.photoOutput to display.photoInput;
    }
}