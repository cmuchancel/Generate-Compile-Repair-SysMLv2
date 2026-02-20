package AutomotiveSystem {
    private import ScalarValues::*;

    part def Rim {
        attribute diameter_in : Real = 18.0;
    }

    part def Tire {
        attribute width_mm : Real = 245.0;
        attribute height_mm : Real = 45.0;
        attribute tread_depth_mm : Real = 6.0;
        attribute min_tread_depth_mm : Real = 3.5;
    }

    part def Wheel {
        attribute rim_diameter_in : Real = 18.0;
        attribute tire_width_mm : Real = 245.0;
        attribute tire_height_mm : Real = 45.0;
        attribute rim_diameter_mm : Real = 457.2;
        attribute outer_diameter_mm : Real = 547.2;

        part rim : Rim;
        part tire : Tire;
    }

    part def BrakeDisc {
        attribute radius_mm : Real = 95.0;
        attribute diameter_mm : Real = 190.0;
    }

    part def WheelAssembly {
        attribute wheel_outer_diameter_mm : Real = 547.2;
        attribute brake_disc_diameter_mm : Real = 190.0;
        attribute fitment_ok : Boolean = true;

        part wheel : Wheel;
        part brakeDisc : BrakeDisc;
    }

    part def Vehicle {
        attribute total_mass_kg : Real = 1200.0;
        attribute length_m : Real = 4.82;

        part frontLeft : WheelAssembly;
        part frontRight : WheelAssembly;
        part rearLeft : WheelAssembly;
        part rearRight : WheelAssembly;
    }

    part vehicle : Vehicle;
}