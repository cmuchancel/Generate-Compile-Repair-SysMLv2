package PhotoCaptureSystem {
    private import ScalarValues::*;

    item def Scene;
    item def Image;
    item def Photo;

    enum def FocusQuality {
        enum WELL_FOCUSED;
        enum NOT_FOCUSED;
    }

    port def SceneInput {
        attribute incoming : Scene;
    }

    port def ImageOutput {
        attribute outgoing : Image;
    }

    port def ImageInput {
        attribute incoming : Image;
    }

    port def PhotoOutput {
        attribute outgoing : Photo;
    }

    port def EvaluationInput {
        attribute incoming : Image;
    }

    port def EvaluationOutput {
        attribute result : FocusQuality;
    }

    part def FocusSystem {
        port input : SceneInput;
        port output : ImageOutput;
    }

    part def ImageEvaluator {
        port input : EvaluationInput;
        port output : EvaluationOutput;
    }

    part def ShootingSystem {
        port input : ImageInput;
        port output : PhotoOutput;
    }

    part def PhotoCaptureController {
        part focus : FocusSystem;
        part evaluator : ImageEvaluator;
        part shooter : ShootingSystem;
        
        port systemInput : SceneInput;
        port systemOutput : PhotoOutput;
        
        connect systemInput to focus.input;
        connect focus.output to evaluator.input;
        connect evaluator.output to shooter.input;
        connect shooter.output to systemOutput;
    }
}