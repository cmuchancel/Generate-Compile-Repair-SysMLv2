package HeadlampTiltAdjustment {
    private import ScalarValues::*;

    enum def TiltStatus {
        enum Valid;
        enum Invalid;
    }

    part def HeadlampTiltKnob {
        attribute minTiltAngle : Real = 50.0;
        attribute maxTiltAngle : Real = 80.0;
        attribute tiltAngle : Real = 65.0;
        attribute pi : Real = 3.14159265358979;
        attribute eulerNumber : Real = 2.71828182845905;
        attribute status : TiltStatus = TiltStatus::Valid;
    }

    part def HeadlampSystem {
        attribute systemName : String = "Headlamp Beam Tilt Adjustment System";
        attribute unitSystem : String = "SI";

        part knob : HeadlampTiltKnob;
    }

    part headlampSystem : HeadlampSystem;
}