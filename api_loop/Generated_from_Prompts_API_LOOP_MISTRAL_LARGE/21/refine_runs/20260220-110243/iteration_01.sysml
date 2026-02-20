package AutomatedPhotographySystem {
    private import ScalarValues::*;

    enum def FocusState {
        enum UNFOCUSED;
        enum FOCUSING;
        enum FOCUSED;
    }

    enum def ShootingState {
        enum IDLE;
        enum PROCESSING;
        enum COMPLETED;
    }

    part def Scene {
        attribute description : String;
    }

    part def FocusProcessor {
        attribute currentState : FocusState = FocusState::UNFOCUSED;
        port focusInput : Scene;
        port focusOutput : Image;
    }

    part def Shooter {
        attribute currentState : ShootingState = ShootingState::IDLE;
        port shootingInput : Image;
        port shootingOutput : Photograph;
    }

    part def Image;
    part def Photograph;

    part system :> FocusProcessor, Shooter {
        part scene : Scene;
        part focusProcessor : FocusProcessor;
        part shooter : Shooter;
        part finalPhotograph : Photograph;

        connect scene to focusProcessor.focusInput;
        connect focusProcessor.focusOutput to shooter.shootingInput;
        connect shooter.shootingOutput to finalPhotograph;
    }
}