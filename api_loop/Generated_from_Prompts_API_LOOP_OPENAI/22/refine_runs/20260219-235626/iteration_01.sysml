package PhotographyWorkflowPackage {
   private import ScalarValues::*;

   item def Scene;
   item def Image;
   item def Picture;

   part def Focus {
      port sceneIn : Scene;
      port imageOut : Image;
   }

   part def Shoot {
      port imageIn : Image;
      port pictureOut : Picture;
   }

   part def PhotographyWorkflow {
      port sceneIn : Scene;
      port pictureOut : Picture;
      part focus : Focus;
      part shoot : Shoot;

      connect sceneIn to focus.sceneIn;
      connect focus.imageOut to shoot.imageIn;
      connect shoot.pictureOut to pictureOut;
   }
}