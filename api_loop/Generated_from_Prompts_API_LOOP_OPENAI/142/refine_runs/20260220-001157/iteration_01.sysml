package HeadlampTiltAdjustment {
    private import ScalarValues::*;
    
    part def HeadlampTiltAdjustmentSystem {
        attribute tiltAngle : Real;
        attribute minTiltAngle : Real = 50.0;
        attribute maxTiltAngle : Real = 80.0;
        attribute piConst : Real = 3.141592653589793;
        attribute eConst : Real = 2.718281828459045;
        attribute withinRange : Boolean = (tiltAngle >= minTiltAngle) and (tiltAngle <= maxTiltAngle);
    }
    
    part headlampTiltAdjustmentSystem : HeadlampTiltAdjustmentSystem;
}