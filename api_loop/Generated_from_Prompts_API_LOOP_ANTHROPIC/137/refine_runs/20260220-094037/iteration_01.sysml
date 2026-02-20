package AutomotiveComponentModel {
    private import ScalarValues::*;

    enum def LengthUnit {
        enum SI;
        enum Imperial;
    }

    part def Tire {
        attribute tireHeight : Real;
    }

    part def Wheel {
        attribute hubDiameter_in : Real = 18.0;
        attribute width_mm : Real = 245.0;
        attribute hubDiameter_mm : Real = 457.2;
        attribute outerDiameter_mm : Real;
        part tire : Tire;
    }

    part def FrontAxle {
        part leftWheel : Wheel;
        part rightWheel : Wheel;
    }

    part def Vehicle {
        attribute mass_kg : Real = 1200.0;
        attribute length_m : Real = 4.82;
        attribute mass_lb : Real = 2645.55;
        attribute length_ft : Real = 15.81;
        part frontAxle : FrontAxle;
    }

    part vehicle : Vehicle {
        attribute redefines mass_kg = 1200.0;
        attribute redefines length_m = 4.82;
        attribute redefines mass_lb = 2645.55;
        attribute redefines length_ft = 15.81;

        part redefines frontAxle : FrontAxle {
            part redefines leftWheel : Wheel {
                attribute redefines hubDiameter_in = 18.0;
                attribute redefines width_mm = 245.0;
                attribute redefines hubDiameter_mm = 457.2;
                attribute redefines outerDiameter_mm = 457.2 + 2.0 * 85.0;
                part redefines tire : Tire {
                    attribute redefines tireHeight = 85.0;
                }
            }
            part redefines rightWheel : Wheel {
                attribute redefines hubDiameter_in = 18.0;
                attribute redefines width_mm = 245.0;
                attribute redefines hubDiameter_mm = 457.2;
                attribute redefines outerDiameter_mm = 457.2 + 2.0 * 85.0;
                part redefines tire : Tire {
                    attribute redefines tireHeight = 85.0;
                }
            }
        }
    }
}