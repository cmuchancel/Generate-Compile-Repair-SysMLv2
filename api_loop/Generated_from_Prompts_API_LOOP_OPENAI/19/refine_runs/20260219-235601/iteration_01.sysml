package PhotographyWorkflow {
  private import ScalarValues::*;

  enum def WorkflowStep {
    enum Selecting;
    enum Focusing;
    enum Shooting;
    enum Completed;
  }

  item def Scene;
  item def FocusedImage;
  item def FinalPhoto;

  port def ScenePort {
    item scene : Scene;
  }

  port def FocusedImagePort {
    item focused : FocusedImage;
  }

  port def PhotoPort {
    item photo : FinalPhoto;
  }

  part def SceneSelector {
    port outScene : ScenePort;
  }

  part def FocusingModule {
    port inScene : ScenePort;
    port outFocused : FocusedImagePort;
    attribute autoFocusEnabled : Boolean = true;
  }

  part def ShootingModule {
    port inFocused : FocusedImagePort;
    port outPhoto : PhotoPort;
  }

  part def CameraSystem {
    part selector : SceneSelector;
    part focusing : FocusingModule;
    part shooting : ShootingModule;
    port outPhoto : PhotoPort;
    attribute currentStep : WorkflowStep = WorkflowStep::Selecting;

    connect selector.outScene to focusing.inScene;
    connect focusing.outFocused to shooting.inFocused;
    connect shooting.outPhoto to outPhoto;
  }

  part system : CameraSystem;
}