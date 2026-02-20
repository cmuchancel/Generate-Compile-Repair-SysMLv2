package VehicleVariants {
    private import ScalarValues::*;

    part def Tire {
        attribute mass : Real;
    }

    part def Axle {
        attribute mass : Real;
        part leftTire : Tire[1];
        part rightTire : Tire[1];
    }

    part def SteeringAxle :> Axle {
        attribute steeringAngle : Real;
    }

    part def Vehicle {
        attribute mass : Real = 1750;
        part frontAxle : Axle[1];
        part rearAxle : Axle[1];
    }

    part def vehicle1_c1 :> Vehicle {
        attribute mass : Real = 2000;
        part frontAxle : SteeringAxle[1];
    }
}