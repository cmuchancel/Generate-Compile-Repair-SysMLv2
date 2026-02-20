package CameraWorkflow {
  private import ScalarValues::*;

  item def Scene;
  item def Image;
  item def Picture;

  port def ScenePort {
    ref item scene : Scene;
  }

  port def ImagePort {
    ref item image : Image;
  }

  port def PicturePort {
    ref item picture : Picture;
  }

  port def DisplayPort {
    ref item picture : Picture;
  }

  part def User {
    port viewPort : ScenePort;
  }

  part def Focusing {
    port sceneIn : ScenePort;
    port imageOut : ImagePort;
  }

  part def Capturing {
    port imageIn : ImagePort;
    port pictureOut : PicturePort;
  }

  part def Displaying {
    port pictureIn : PicturePort;
    port displayPort : DisplayPort;
  }

  part def CameraSystem {
    port viewPort : ScenePort;
    port displayPort : DisplayPort;

    part focusing : Focusing;
    part capturing : Capturing;
    part displaying : Displaying;

    connect viewPort to focusing.sceneIn;
    connect focusing.imageOut to capturing.imageIn;
    connect capturing.pictureOut to displaying.pictureIn;
    connect displaying.displayPort to displayPort;
  }

  part def SystemContext {
    part user : User;
    part camera : CameraSystem;
    connect user.viewPort to camera.viewPort;
  }
}