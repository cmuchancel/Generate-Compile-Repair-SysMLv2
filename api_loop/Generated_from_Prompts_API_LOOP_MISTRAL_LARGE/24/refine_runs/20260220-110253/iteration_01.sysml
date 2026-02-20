package PhotoWorkflow {

    private import ScalarValues::*;

    enum def FocusQuality {
        enum WELL_FOCUSED;
        enum POORLY_FOCUSED;
    }

    part def Scene {
        attribute description : String;
    }

    part def FocusEvaluator {
        attribute focusQuality : FocusQuality;
    }

    part def Image {
        attribute isFocused : Boolean;
    }

    part def Photograph {
        attribute filePath : String;
    }

    part def CameraSystem {
        part scene : Scene;
        part focusEvaluator : FocusEvaluator;
        part image : Image;
        part photograph : Photograph;

        port def FocusInput;
        port def FocusOutput;
        port def ImageInput;
        port def ImageOutput;
        port def PhotoOutput;

        port focusIn : FocusInput;
        port focusOut : FocusOutput;
        port imageIn : ImageInput;
        port imageOut : ImageOutput;
        port photoOut : PhotoOutput;
    }

    part workflow : CameraSystem;

    connect workflow.focusOut to workflow.focusEvaluator;
    connect workflow.imageOut to workflow.image;
    connect workflow.image to workflow.photoOut;
}