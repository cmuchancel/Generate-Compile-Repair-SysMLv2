package CameraShootingProcess {
  private import ScalarValues::*;

  enum def SceneType { Auto; Portrait; Landscape; Night; }

  item def SceneSelection;
  item def FocusedScene;
  item def RawImage;
  item def FinalPhoto;

  port def ScenePort;
  port def FocusPort;
  port def ImagePort;

  part def UserInterface {
    port selectOut : ScenePort;
  }

  part def SceneSelector {
    attribute selectedScene : SceneType;
    port selectIn : ScenePort;
    port sceneOut : ScenePort;
  }

  part def AutoFocus {
    attribute autoFocusEnabled : Boolean = true;
    port sceneIn : ScenePort;
    port focusedOut : FocusPort;
  }

  part def ImageGenerator {
    attribute autoGenerateEnabled : Boolean = true;
    port focusedIn : FocusPort;
    port imageOut : ImagePort;
  }

  part def ImagingModule {
    attribute autoCaptureEnabled : Boolean = true;
    port imageIn : ImagePort;
    port photoOut : ImagePort;
  }

  part def CameraSystem {
    part ui : UserInterface;
    part selector : SceneSelector;
    part af : AutoFocus;
    part generator : ImageGenerator;
    part imager : ImagingModule;

    port photoOut : ImagePort;

    connect ui.selectOut to selector.selectIn;
    connect selector.sceneOut to af.sceneIn;
    connect af.focusedOut to generator.focusedIn;
    connect generator.imageOut to imager.imageIn;
    connect imager.photoOut to photoOut;
  }
}