package CameraShootingSystem {
    private import ScalarValues::*;

    enum def ShootingScene {
        enum Landscape;
        enum Portrait;
        enum Night;
        enum Sports;
        enum Macro;
    }

    part def Viewfinder {
        attribute isActive : Boolean = false;
    }

    part def FocusModule {
        attribute isFocused : Boolean = false;
        attribute selectedScene : ShootingScene;
    }

    part def ImagingModule {
        attribute isCapturing : Boolean = false;
        attribute finalPhoto : String;
    }

    part def CameraController {
        port controlPort : ControlPort;
    }

    port def ControlPort;

    part cameraSystem {
        part viewfinder : Viewfinder;
        part focusModule : FocusModule;
        part imagingModule : ImagingModule;
        part controller : CameraController;

        connect controller.controlPort to focusModule;
        connect controller.controlPort to imagingModule;
    }
}