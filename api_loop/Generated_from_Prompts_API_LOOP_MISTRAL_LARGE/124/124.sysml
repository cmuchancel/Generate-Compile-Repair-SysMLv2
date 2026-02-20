package VehicleSystem {
    private import ScalarValues::*;

    part def Vehicle {
        attribute totalMass : Real = 2500.0;
        part frontAxleAssembly : FrontAxleAssembly;
        part rearAxleAssembly : RearAxleAssembly;
    }

    part def FrontAxleAssembly {
        attribute mass : Real = 150.0;
        part frontAxle : FrontAxle;
        part frontWheel1 : Wheel;
        part frontWheel2 : Wheel;
    }

    part def FrontAxle {
        attribute mass : Real;
        attribute steeringAngle : Real;
    }

    part def RearAxleAssembly {
        attribute mass : Real = 250.0;
        part rearAxle : RearAxle;
        part rearWheel1 : Wheel;
        part rearWheel2 : Wheel;
    }

    part def RearAxle {
        attribute mass : Real;
    }

    part def Wheel {
    }

    part vehicle : Vehicle;
}