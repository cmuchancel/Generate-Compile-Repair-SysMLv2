package AutomotiveModel
    private import ScalarValues::*;

    part def Vehicle {
        attribute mass_kg : Real = 1200;
        attribute mass_lb : Real = mass_kg * 2.20462;
        attribute length_m : Real = 4.82;
        attribute length_ft : Real = length_m * 3.28084;
        part frontLeftWheel : Wheel;
        part frontRightWheel : Wheel;
    }

    part def Wheel {
        attribute hubDiameter_in : Real = 18;
        attribute hubDiameter_mm : Real = hubDiameter_in * 25.4;
        attribute width_mm : Real = 245;
        attribute width_in : Real = width_mm / 25.4;
        part tire : Tire;
        attribute outerDiameter_mm : Real = hubDiameter_mm + 2 * tire.height_mm;
        attribute outerDiameter_in : Real = outerDiameter_mm / 25.4;
    }

    part def Tire {
        attribute height_mm : Real = 0;
        attribute height_in : Real = height_mm / 25.4;
    }

    part vehicle : Vehicle;
end;