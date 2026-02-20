package PhotographyWorkflow {
  private import ScalarValues::*;

  item def Scene;
  item def Image;
  item def Photo;
  item def TakePhotoCommand;

  port def ScenePort;
  port def ImagePort;
  port def PhotoPort;
  port def CommandPort;

  part def SceneAcquirer {
    port triggerIn : CommandPort;
    port sceneOut : ScenePort;
  }

  part def FocusFunction {
    port sceneIn : ScenePort;
    port imageOut : ImagePort;
  }

  part def ShootingFunction {
    port imageIn : ImagePort;
    port photoOut : PhotoPort;
  }

  part def PhotoDisplay {
    port photoIn : PhotoPort;
  }

  part def PhotographySystem {
    port takePhoto : CommandPort;
    part sceneAcquirer : SceneAcquirer;
    part focusFunction : FocusFunction;
    part shootingFunction : ShootingFunction;
    part photoDisplay : PhotoDisplay;

    connect takePhoto to sceneAcquirer.triggerIn;
    connect sceneAcquirer.sceneOut to focusFunction.sceneIn;
    connect focusFunction.imageOut to shootingFunction.imageIn;
    connect shootingFunction.photoOut to photoDisplay.photoIn;
  }

  part system : PhotographySystem;
}