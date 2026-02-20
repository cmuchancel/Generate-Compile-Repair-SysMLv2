package VehicleConfiguration {

    private import ScalarValues::*;

    part def Vehicle {
        attribute mass : Real = 1750.0;
        part frontAxle : Axle;
        part rearAxle : Axle;
    }

    part def Axle {
        attribute mass : Real;
        part tires : Tire[2];
    }

    part def Tire {
        attribute mass : Real;
    }

    part def FrontAxle :> Axle {
        attribute steeringAngle : Real;
    }

    part vehicle1 : Vehicle {
        mass = 2000.0;
        part frontAxle : FrontAxle {
            steeringAngle = 30.0;
        }
        part rearAxle : Axle;
    }

    part vehicle1_c1 : vehicle1 {
        part frontAxle : FrontAxle {
            steeringAngle = 35.0;
        }
        part rearAxle : Axle {
            part tires : Tire {
                :>> tires[1] { mass = 15.0; }
                :>> tires[2] { mass = 15.0; }
            }
        }
    }
}