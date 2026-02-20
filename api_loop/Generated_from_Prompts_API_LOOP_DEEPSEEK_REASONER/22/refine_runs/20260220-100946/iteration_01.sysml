package PhotographyWorkflow {
    private import ScalarValues::*;
    
    item def Scene {
        attribute description : String;
    }
    
    item def Image {
        attribute data : String;
    }
    
    item def Picture {
        attribute data : String;
    }
    
    port def SceneInput {
        attribute scene : Scene;
    }
    
    port def ImageOutput {
        attribute image : Image;
    }
    
    port def PictureOutput {
        attribute picture : Picture;
    }
    
    part def Focus {
        port input : SceneInput;
        port output : ImageOutput;
        
        attribute operation : String = "Focus";
    }
    
    part def Shoot {
        port input : ImageOutput;
        port output : PictureOutput;
        
        attribute operation : String = "Shoot";
    }
    
    part def PhotographySystem {
        part focusUnit : Focus;
        part shootUnit : Shoot;
        port userInput : SceneInput;
        port finalOutput : PictureOutput;
        
        connect userInput to focusUnit.input;
        connect focusUnit.output to shootUnit.input;
        connect shootUnit.output to finalOutput;
    }
}