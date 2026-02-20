package CameraSystemModel {
  private import ScalarValues::*;

  enum def SceneType {
    enum portrait;
    enum landscape;
    enum night;
    enum macro;
    enum other;
  }

  item def SceneSelection {
    attribute sceneType : SceneType;
    attribute confirmed : Boolean = true;
  }

  port def ScenePort {
    attribute selection : SceneSelection;
  }

  item def Image {
    attribute clear : Boolean = true;
    attribute realistic : Boolean = true;
  }

  port def ImagePort {
    attribute image : Image;
  }

  item def Photo {
    attribute image : Image;
    attribute recorded : Boolean = true;
  }

  port def PhotoPort {
    attribute photo : Photo;
  }

  part def AutofocusSubsystem {
    attribute autoFocusEnabled : Boolean = true;
    attribute focusComplete : Boolean = true;
    port sceneIn : ScenePort;
    port imageOut : ImagePort;
  }

  part def ImagingSubsystem {
    attribute autoImagingEnabled : Boolean = true;
    attribute captureComplete : Boolean = true;
    port imageIn : ImagePort;
    port photoOut : PhotoPort;
  }

  part def StorageSubsystem {
    attribute autoSaveEnabled : Boolean = true;
    attribute savedCount : Integer = 0;
    port photoIn : PhotoPort;
  }

  part def UserInterface {
    attribute userSelectsScene : Boolean = true;
    port sceneOut : ScenePort;
  }

  part def CameraSystem {
    attribute automaticOperation : Boolean = true;
    part ui : UserInterface;
    part autofocus : AutofocusSubsystem;
    part imaging : ImagingSubsystem;
    part storage : StorageSubsystem;
    connect ui.sceneOut to autofocus.sceneIn;
    connect autofocus.imageOut to imaging.imageIn;
    connect imaging.photoOut to storage.photoIn;
  }
}