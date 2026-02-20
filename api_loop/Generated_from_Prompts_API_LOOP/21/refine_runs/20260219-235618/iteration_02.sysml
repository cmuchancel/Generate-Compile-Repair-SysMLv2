package PhotographyAutomation {
  private import ScalarValues::*;

  item def Scene;
  item def FocusedImage;
  item def Photo;

  port def ScenePort {
    ref item scene : Scene;
  }

  port def FocusedImagePort {
    ref item image : FocusedImage;
  }

  port def PhotoPort {
    ref item photo : Photo;
  }

  part def User {
    port sceneOut : ScenePort;
    port photoIn : PhotoPort;
  }

  part def FocusProcessor {
    port sceneIn : ScenePort;
    port imageOut : FocusedImagePort;
  }

  part def ShootingOperator {
    port imageIn : FocusedImagePort;
    port photoOut : PhotoPort;
  }

  part def PhotographySystem {
    port sceneIn : ScenePort;
    port photoOut : PhotoPort;
    part focus : FocusProcessor;
    part shoot : ShootingOperator;
    connect sceneIn to focus.sceneIn;
    connect focus.imageOut to shoot.imageIn;
    connect shoot.photoOut to photoOut;
  }

  part def PhotographyEnvironment {
    part user : User;
    part system : PhotographySystem;
    connect user.sceneOut to system.sceneIn;
    connect system.photoOut to user.photoIn;
  }
}