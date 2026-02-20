package HeadlampTiltAdjustment {
    private import ScalarValues::*;

    enum def TiltAngleStatus {
        enum IN_RANGE;
        enum BELOW_MIN;
        enum ABOVE_MAX;
    }

    part def HeadlampTiltKnob {
        attribute minAngle : Real = 50.0;
        attribute maxAngle : Real = 80.0;
        attribute currentAngle : Real;
        attribute status : TiltAngleStatus;
    }

    part def TiltAngleValidator {
        port validate : ValidationPort;
    }

    port def ValidationPort {
        in angle : Real;
        out status : TiltAngleStatus;
    }

    part tiltKnob : HeadlampTiltKnob;
    part validator : TiltAngleValidator;

    connect tiltKnob.currentAngle to validator.validate.angle;
    connect validator.validate.status to tiltKnob.status;
}