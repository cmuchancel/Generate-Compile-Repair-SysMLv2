package PhotoAcquisitionSystem {
   private import ScalarValues::*;

   item def Scene;
   item def Image;
   item def Photo;

   port def SceneSelectionPort {
      attribute scene : Scene;
   }

   port def ImagePort {
      attribute image : Image;
   }

   port def PhotoPort {
      attribute photo : Photo;
   }

   part def User {
      port selectionOut : SceneSelectionPort;
      port photoIn : PhotoPort;
   }

   part def FocusingModule {
      port sceneIn : SceneSelectionPort;
      port focusedImageOut : ImagePort;
   }

   part def CaptureModule {
      port focusedImageIn : ImagePort;
      port photoOut : PhotoPort;
   }

   part def CameraSystem {
      part focusing : FocusingModule;
      part capture : CaptureModule;
      port sceneIn : SceneSelectionPort;
      port photoOut : PhotoPort;
      attribute focusedImage : Image;
      attribute finalPhoto : Photo;

      connect sceneIn to focusing.sceneIn;
      connect focusing.focusedImageOut to capture.focusedImageIn;
      connect capture.photoOut to photoOut;
   }

   part def PhotoService {
      part user : User;
      part camera : CameraSystem;

      connect user.selectionOut to camera.sceneIn;
      connect camera.photoOut to user.photoIn;
   }

   part system : PhotoService;
}