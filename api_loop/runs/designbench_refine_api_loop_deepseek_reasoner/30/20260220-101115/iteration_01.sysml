package DigitalCameraSystem {
    private import ScalarValues::*;

    enum def FocusStatus {
        enum NotFocused;
        enum Focused;
    }

    part def Viewfinder {
        attribute selectedScene : String;
        port output sceneInfo : out String;
    }

    part def Autofocus {
        port input targetScene : in String;
        port output focusComplete : out FocusStatus;
        attribute currentFocus : FocusStatus = FocusStatus::NotFocused;
    }

    part def ImagingModule {
        port input sceneData : in String;
        port input focusReady : in FocusStatus;
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