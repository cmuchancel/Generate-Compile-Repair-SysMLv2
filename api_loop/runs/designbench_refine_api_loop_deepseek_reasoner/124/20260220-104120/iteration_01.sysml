package VehicleSystem {
    private import ScalarValues::*;

    part def Car {
        attribute totalMass : Real = 2500.0 kg;
        part frontAxleAssembly : FrontAxleAssembly;
        part rearAxleAssembly : RearAxleAssembly;
    }

    part def FrontAxleAssembly {
        attribute mass : Real = 150.0 kg;
        part frontAxle : FrontAxle;
        part frontWheel1 : FrontWheel;
        part frontWheel2 : FrontWheel;
    }

    part def FrontAxle {
        attribute mass : Real;
        attribute steeringAngle : Real deg;
    }

    part def FrontWheel {
        attribute mass : Real;
    }

    part def RearAxleAssembly {
        attribute mass : Real = 250.0 kg;
        part rearAxle : RearAxle;
        part rearWheel1 : RearWheel;
        part rearWheel2 : RearWheel;
    }

    part def RearAxle {
        attribute mass : Real;
    }

    part def RearWheel {
        attribute mass : Real;
    }
}