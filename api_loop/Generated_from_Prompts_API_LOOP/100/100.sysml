package VehicleConfig {
    private import ScalarValues::*;

    part def Tire {
        attribute mass : Real = 20;
    }

    part def FrontAxle {
        attribute mass : Real = 300;
        part leftTire : Tire;
        part rightTire : Tire;
    }

    part def RearAxle {
        attribute mass : Real = 320;
        part leftTire : Tire;
        part rightTire : Tire;
    }

    part def DedicatedFrontAxle specializes FrontAxle {
        attribute steeringAngle : Real = 0;
    }

    part def FrontAxle_C1 specializes DedicatedFrontAxle {
        part tireFL : Tire redefines leftTire;
        part tireFR : Tire redefines rightTire;
    }

    part def RearAxle_C1 specializes RearAxle {
        part tireRL : Tire redefines leftTire;
        part tireRR : Tire redefines rightTire;
    }

    part def Vehicle {
        attribute mass : Real = 1750;
        part frontAxle : FrontAxle;
        part rearAxle : RearAxle;
    }

    part def Vehicle1_C1 specializes Vehicle {
        attribute mass : Real = 2000;
        part frontAxle_c1 : FrontAxle_C1 redefines frontAxle;
        part rearAxle_c1 : RearAxle_C1 redefines rearAxle;
    }
}