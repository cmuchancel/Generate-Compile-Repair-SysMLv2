package DigitalCameraSystem {
    private import ScalarValues::*;

    enum def FocusStatus {
        enum NotFocused;
        enum Focused;
    }

    part def Viewfinder {
        attribute selectedScene : String;
        port sceneInfo;
    }

    part def Autofocus {
        port targetScene;
        port focusComplete;
        attribute currentFocus : FocusStatus = FocusStatus::NotFocused;
    }

    part def ImagingModule {
        port sceneData;
        port focusReady;
        attribute photoReady : Boolean = false;
    }

    part def Camera {
        part viewfinder : Viewfinder;
        part autofocus : Autofocus;
        part imagingModule : ImagingModule;

        connect viewfinder.sceneInfo to autofocus.targetScene;
        connect autofocus.focusComplete to imagingModule.focusReady;
        connect viewfinder.sceneInfo to imagingModule.sceneData;
    }
}