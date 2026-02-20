package PhotoSystem {
    private import ScalarValues::*;
    
    item def Image {
        attribute id : String;
        attribute isFocused : Boolean;
    }
    
    port def ImageInputPort {
        attribute image : Image;
    }
    
    port def ImageOutputPort {
        attribute image : Image;
    }
    
    part def FocusModule {
        port input : in ImageInputPort;
        port output : out ImageOutputPort;
        attribute intermediateImage : Image;
    }
    
    part def CaptureModule {
        port input : in ImageInputPort;
        port output : out ImageOutputPort;
        attribute finalPhoto : Image;
    }
    
    part def User {
        port sceneSelection : out ImageOutputPort;
        port photoDelivery : in ImageInputPort;
    }
    
    part def CameraSystem {
        part focusModule : FocusModule;
        part captureModule : CaptureModule;
        port systemInput : in ImageInputPort;
        port systemOutput : out ImageOutputPort;
        
        connect systemInput to focusModule.input;
        connect focusModule.output to captureModule.input;
        connect captureModule.output to systemOutput;
    }
    
    part user : User;
    part cameraSystem : CameraSystem;
    
    connect user.sceneSelection to cameraSystem.systemInput;
    connect cameraSystem.systemOutput to user.photoDelivery;
}