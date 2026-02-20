package VehicleStructuralArchitecture {
    private import ScalarValues::*;

    part def Vehicle {
        attribute mass : Real;
        part frontAxleAssembly : AxleAssembly;
        part rearAxleAssembly : AxleAssembly;
    }

    part def AxleAssembly {
        part axle : Axle;
        part wheel1 : Wheel;
        part wheel2 : Wheel;
    }

    part def Axle {
    }

    part def FrontAxle specializes Axle {
        attribute steeringAngle : Real;
    }

    part def Wheel {
        attribute name : String;
    }

    part vehicle1 : Vehicle {
        mass = 1750.0;
        frontAxleAssembly = {
            part axle = frontAxle1 : FrontAxle;
            part wheel1 = leftFrontWheel : Wheel;
            part wheel2 = rightFrontWheel : Wheel;
        };
        rearAxleAssembly = {
            part axle = rearAxle1 : Axle;
            part wheel1 = leftRearWheel : Wheel;
            part wheel2 = rightRearWheel : Wheel;
        };
    }

    frontAxle1.steeringAngle = 0.0;
    leftFrontWheel.name = "Left Front Wheel";
    rightFrontWheel.name = "Right Front Wheel";
    leftRearWheel.name = "Left Rear Wheel";
    rightRearWheel.name = "Right Rear Wheel";

    part vehicle2 : Vehicle {
        mass = 2000.0;
        frontAxleAssembly = {
            part axle = frontAxle2 : FrontAxle;
            part wheel1 = frontWheel1 : Wheel;
            part wheel2 = frontWheel2 : Wheel;
        };
        rearAxleAssembly = {
            part axle = rearAxle2 : Axle;
            part wheel1 = rearWheel1 : Wheel;
            part wheel2 = rearWheel2 : Wheel;
        };
    }

    frontAxle2.steeringAngle = 0.0;
}