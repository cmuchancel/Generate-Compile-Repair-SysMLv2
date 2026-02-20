package DigitalCameraSystem {
    private import ScalarValues::*;

    enum def FocusStatus {
        enum NotFocused;
        enum Focused;
    }

    part def Viewfinder {
        attribute selectedScene : String;
        port sceneInfo : out;
    }

    part def Autofocus {
        port targetScene : in;
        port focusComplete : out;
        attribute currentFocus : FocusStatus = FocusStatus::NotFocused;
    }

    part def ImagingModule {
        port sceneData : in;
        port focusReady : in;
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