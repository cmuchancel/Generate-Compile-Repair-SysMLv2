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
            axle = frontAxle1 : FrontAxle {
                steeringAngle = 0.0;
            },
            wheel1 = leftFrontWheel : Wheel {
                name = "Left Front Wheel";
            },
            wheel2 = rightFrontWheel : Wheel {
                name = "Right Front Wheel";
            }
        };
        rearAxleAssembly = {
            axle = rearAxle1 : Axle {},
            wheel1 = leftRearWheel : Wheel {
                name = "Left Rear Wheel";
            },
            wheel2 = rightRearWheel : Wheel {
                name = "Right Rear Wheel";
            }
        };
    }

    part vehicle2 : Vehicle {
        mass = 2000.0;
        frontAxleAssembly = {
            axle = frontAxle2 : FrontAxle {
                steeringAngle = 0.0;
            },
            wheel1 = frontWheel1 : Wheel {},
            wheel2 = frontWheel2 : Wheel {}
        };
        rearAxleAssembly = {
            axle = rearAxle2 : Axle {},
            wheel1 = rearWheel1 : Wheel {},
            wheel2 = rearWheel2 : Wheel {}
        };
    }
}