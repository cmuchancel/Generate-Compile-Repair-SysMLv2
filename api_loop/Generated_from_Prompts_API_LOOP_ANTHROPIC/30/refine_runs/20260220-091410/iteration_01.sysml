package DigitalCameraShootingProcess {
    private import ScalarValues::*;

    enum def ShootingStatus {
        enum IDLE;
        enum VIEWFINDING;
        enum FOCUSING;
        enum IMAGING;
        enum COMPLETE;
    }

    enum def FocusStatus {
        enum UNFOCUSED;
        enum FOCUSING;
        enum FOCUSED;
    }

    enum def ImageQuality {
        enum LOW;
        enum MEDIUM;
        enum HIGH;
    }

    item def Scene {
        attribute sceneName : String;
        attribute sceneDescription : String;
    }

    item def FocusData {
        attribute focusDistance : Real;
        attribute focusAccuracy : Real;
        attribute status : FocusStatus;
    }

    item def ImageData {
        attribute width : Integer;
        attribute height : Integer;
        attribute quality : ImageQuality;
        attribute isCapture : Boolean;
    }

    item def Photo {
        attribute width : Integer;
        attribute height : Integer;
        attribute quality : ImageQuality;
        attribute fileName : String;
    }

    port def SceneInputPort {
        attribute sceneReceived : Boolean;
    }

    port def FocusDataPort {
        attribute focusReady : Boolean;
    }

    port def ImageDataPort {
        attribute imageReady : Boolean;
    }

    port def PhotoOutputPort {
        attribute photoProduced : Boolean;
    }

    part def ViewfindingModule {
        attribute isActive : Boolean;
        attribute currentStatus : ShootingStatus;
        attribute selectedScene : String;

        port sceneInput : SceneInputPort;
        port focusDataOut : FocusDataPort;
    }

    part def FocusingModule {
        attribute isActive : Boolean;
        attribute focusStatus : FocusStatus;
        attribute autoFocusEnabled : Boolean;
        attribute focusDistance : Real;

        port focusDataIn : FocusDataPort;
        port imageDataOut : ImageDataPort;
    }

    part def ImagingModule {
        attribute isActive : Boolean;
        attribute captureInitiated : Boolean;
        attribute autoCapture : Boolean;
        attribute imageQuality : ImageQuality;

        port imageDataIn : ImageDataPort;
        port photoOut : PhotoOutputPort;
    }

    part def UserInterface {
        attribute userActionRequired : Boolean;
        attribute sceneSelected : Boolean;
        attribute selectedScene : String;

        port sceneOut : SceneInputPort;
    }

    part def PhotoStorage {
        attribute storageAvailable : Boolean;
        attribute photosStored : Integer;

        port photoIn : PhotoOutputPort;
    }

    part def DigitalCamera {
        attribute cameraName : String;
        attribute isFullyAutomated : Boolean;
        attribute shootingStatus : ShootingStatus;

        part userInterface : UserInterface;
        part viewfindingModule : ViewfindingModule;
        part focusingModule : FocusingModule;
        part imagingModule : ImagingModule;
        part photoStorage : PhotoStorage;

        connect userInterface.sceneOut to viewfindingModule.sceneInput;
        connect viewfindingModule.focusDataOut to focusingModule.focusDataIn;
        connect focusingModule.imageDataOut to imagingModule.imageDataIn;
        connect imagingModule.photoOut to photoStorage.photoIn;
    }
}