package PhotographyWorkflow {
  private import ScalarValues::*;

  item def Scene;
  item def Image;
  item def Photograph;

  port def ScenePort {
    ref item scene : Scene;
  }

  port def ImagePort {
    ref item image : Image;
  }

  port def PhotoPort {
    ref item photo : Photograph;
  }

  part def User {
    port sceneSelection : ScenePort;
    port photoIn : PhotoPort;
  }

  part def FocusingProcess {
    port sceneIn : ScenePort;
    port imageOut : ImagePort;
    attribute focused : Boolean = true;
  }

  part def ShootingProcess {
    port imageIn : ImagePort;
    port photoOut : PhotoPort;
    attribute shot : Boolean = true;
  }

  part def WorkflowSystem {
    part focusing : FocusingProcess;
    part shooting : ShootingProcess;
    port sceneIn : ScenePort;
    port photoOut : PhotoPort;
    attribute automaticWorkflow : Boolean = true;
    connect sceneIn to focusing.sceneIn;
    connect focusing.imageOut to shooting.imageIn;
    connect shooting.photoOut to photoOut;
  }

  part def PhotographyWorkflow {
    part user : User;
    part system : WorkflowSystem;
    connect user.sceneSelection to system.sceneIn;
    connect system.photoOut to user.photoIn;
  }

  part workflow : PhotographyWorkflow;
}