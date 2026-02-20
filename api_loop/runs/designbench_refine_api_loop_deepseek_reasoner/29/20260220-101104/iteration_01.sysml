package CameraSystem {
    private import ScalarValues::*;
    
    enum def SceneType {
        enum Landscape;
        enum Portrait;
        enum Night;
        enum Macro;
    }
    
    item def Photo {
        attribute id : String;
        attribute timestamp : String;
        attribute data : String;
    }
    
    part def AutofocusSubsystem {
        attribute isFocused : Boolean = false;
        port focusInput : ScenePort;
        port focusOutput : ScenePort;
    }
    
    part def ImagingSubsystem {
        attribute photosTaken : Integer = 0;
        port imageInput : ScenePort;
        port imageOutput : PhotoPort;
    }
    
    part def StorageSubsystem {
        attribute storedPhotos : Integer = 0;
        port storageInput : PhotoPort;
    }
    
    port def ScenePort {
        attribute currentScene : SceneType;
        attribute isFocused : Boolean;
    }
    
    port def PhotoPort {
        attribute photo : Photo;
    }
    
    part def Camera {
        part autofocus : AutofocusSubsystem;
        part imaging : ImagingSubsystem;
        part storage : StorageSubsystem;
        port userInput : ScenePort;
        port photoOutput : PhotoPort;
        
        connect userInput to autofocus.focusInput;
        connect autofocus.focusOutput to imaging.imageInput;
        connect imaging.imageOutput to storage.storageInput;
        connect imaging.imageOutput to photoOutput;
    }
}