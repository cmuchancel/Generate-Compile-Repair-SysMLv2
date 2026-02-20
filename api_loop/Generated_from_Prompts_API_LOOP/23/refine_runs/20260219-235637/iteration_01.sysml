package PhotographyWorkflow {
  private import ScalarValues::*;

  enum def FocusState {
    enum clear;
    enum unclear;
  }

  item def Scene;
  item def Photo;

  port def ScenePort {
    attribute scene : Scene;
  }

  port def FocusStatusPort {
    attribute status : FocusState;
  }

  port def PhotoPort {
    attribute photo : Photo;
  }

  part def SceneSelector {
    port sceneOut : ScenePort;
  }

  part def FocusUnit {
    attribute autoFocus : Boolean = true;
    port sceneIn : ScenePort;
    port focusStatusOut : FocusStatusPort;
  }

  part def PhotoShooter {
    attribute oneClick : Boolean = true;
    port sceneIn : ScenePort;
    port focusIn : FocusStatusPort;
    port photoOut : PhotoPort;
  }

  part def PhotographySystem {
    part selector : SceneSelector;
    part focus : FocusUnit;
    part shooter : PhotoShooter;

    port photoOut : PhotoPort;

    connect selector.sceneOut to focus.sceneIn;
    connect selector.sceneOut to shooter.sceneIn;
    connect focus.focusStatusOut to shooter.focusIn;
    connect shooter.photoOut to photoOut;
  }

  part system : PhotographySystem;
}