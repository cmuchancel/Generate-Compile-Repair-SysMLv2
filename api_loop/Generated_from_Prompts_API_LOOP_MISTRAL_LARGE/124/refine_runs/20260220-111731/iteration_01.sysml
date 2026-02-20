package VehicleSystem {
    private import ScalarValues::*;

    part def Vehicle {
        attribute totalMass : MassValue = 2500 kg;
        part frontAxleAssembly : FrontAxleAssembly;
        part rearAxleAssembly : RearAxleAssembly;
    }

    part def FrontAxleAssembly {
        attribute mass : MassValue = 150 kg;
        part frontAxle : FrontAxle;
        part frontWheel1 : Wheel;
        part frontWheel2 : Wheel;
    }

    part def FrontAxle {
        attribute mass : MassValue;
        attribute steeringAngle : AngleValue;
    }

    part def RearAxleAssembly {
        attribute mass : MassValue = 250 kg;
        part rearAxle : RearAxle;
        part rearWheel1 : Wheel;
        part rearWheel2 : Wheel;
    }

    part def RearAxle {
        attribute mass : MassValue;
    }

    part def Wheel {
    }

    part vehicle : Vehicle;
}