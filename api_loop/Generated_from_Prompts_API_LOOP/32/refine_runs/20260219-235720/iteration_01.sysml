package CameraSystemModel {
  private import ScalarValues::*;

  item def Scene;
  item def Image;
  item def Photo;

  port def ScenePort;
  port def ImagePort;
  port def PhotoPort;

  part def User {
    port sceneOut : ScenePort;
    port viewIn : PhotoPort;
  }

  part def Focuser {
    port sceneIn : ScenePort;
    port imageOut : ImagePort;
  }

  part def Capturer {
    port imageIn : ImagePort;
    port photoOut : PhotoPort;
  }

  part def Display {
    port photoIn : PhotoPort;
    port photoOut : PhotoPort;
  }

  part def CameraSystem {
    port sceneIn : ScenePort;
    port photoOut : PhotoPort;
    part focuser : Focuser;
    part capturer : Capturer;
    part display : Display;

    connect sceneIn to focuser.sceneIn;
    connect focuser.imageOut to capturer.imageIn;
    connect capturer.photoOut to display.photoIn;
    connect display.photoOut to photoOut;
  }

  part def ShootingExperience {
    part user : User;
    part system : CameraSystem;

    connect user.sceneOut to system.sceneIn;
    connect system.photoOut to user.viewIn;
  }
}