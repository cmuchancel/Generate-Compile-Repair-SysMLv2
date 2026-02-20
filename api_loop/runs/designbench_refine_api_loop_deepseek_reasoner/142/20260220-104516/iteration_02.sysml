package HeadlampTiltSystem {
    private import ScalarValues::*;

    enum def ValidationResult {
        enum VALID;
        enum INVALID;
    }

    part def Headlamp {
        attribute beamTiltAngle : Real;
    }

    part def Validator {
        attribute minAngle : Real = 50.0;
        attribute maxAngle : Real = 80.0;
        attribute result : ValidationResult;
    }

    port def AnglePort;

    part def TiltSystem {
        part headlamp : Headlamp;
        part validator : Validator;
        port headlampPort : AnglePort;
        port validatorPort : AnglePort;

        connect headlampPort to validatorPort;
    }
}