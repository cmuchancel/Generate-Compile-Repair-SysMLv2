package CameraSystem {
    private import ScalarValues::*;

    enum def FocusStatus {
        enum Focusing;
        enum Focused;
        enum Failed;
    }

    enum def ImageStatus {
        enum Capturing;
        enum Captured;
        enum Saved;
    }

    item def Scene {
        attribute sceneName : String;
        attribute sceneDescription : String;
    }

    item def Photo {
        attribute photoId : String;
        attribute resolution : String;
        attribute isSaved : Boolean;
    }

    item def FocusResult {
        attribute status : FocusStatus;
        attribute focusDistance : Real;
        attribute isClear : Boolean;
    }

    item def ImageData {
        attribute status : ImageStatus;
        attribute quality : String;
        attribute isRealistic : Boolean;
    }

    port def SceneInputPort {
        attribute selectedScene : String;
    }

    port def PhotoOutputPort {
        attribute capturedPhoto : String;
        attribute photoSaved : Boolean;
    }

    port def FocusPort {
        attribute focusStatus : String;
        attribute focusComplete : Boolean;
    }

    port def ImagingPort {
        attribute imagingStatus : String;
        attribute imageReady : Boolean;
    }

    part def AutofocusSubsystem {
        attribute currentFocusStatus : FocusStatus;
        attribute focusDistance : Real;
        attribute isObjectDetected : Boolean;
        attribute isClearImageObtained : Boolean;

        port focusInputPort : SceneInputPort;
        port focusOutputPort : FocusPort;
    }

    part def ImagingSubsystem {
        attribute currentImageStatus : ImageStatus;
        attribute imageQuality : String;
        attribute isCapturing : Boolean;

        port imagingInputPort : FocusPort;
        port imagingOutputPort : ImagingPort;
    }

    part def PhotoStorage {
        attribute totalPhotoCount : Integer;
        attribute storageCapacity : Integer;
        attribute storageUsed : Real;
        attribute allPhotosSaved : Boolean;

        port storageInputPort : ImagingPort;
        port storageOutputPort : PhotoOutputPort;
    }

    part def UserInterface {
        attribute isSceneSelected : Boolean;
        attribute selectedSceneName : String;
        attribute isPhotoConfirmed : Boolean;

        port userSceneOutputPort : SceneInputPort;
        port userPhotoInputPort : PhotoOutputPort;
    }

    part def Camera {
        attribute cameraModel : String;
        attribute isOperational : Boolean;
        attribute currentPhotoCount : Integer;

        port cameraSceneInput : SceneInputPort;
        port cameraPhotoOutput : PhotoOutputPort;

        part userInterface : UserInterface;
        part autofocusSubsystem : AutofocusSubsystem;
        part imagingSubsystem : ImagingSubsystem;
        part photoStorage : PhotoStorage;

        connect userInterface.userSceneOutputPort to autofocusSubsystem.focusInputPort;
        connect autofocusSubsystem.focusOutputPort to imagingSubsystem.imagingInputPort;
        connect imagingSubsystem.imagingOutputPort to photoStorage.storageInputPort;
        connect photoStorage.storageOutputPort to userInterface.userPhotoInputPort;
    }
}