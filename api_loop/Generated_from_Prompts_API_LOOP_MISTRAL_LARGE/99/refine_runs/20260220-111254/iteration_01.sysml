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
        attribute isSteerable : Boolean = false;
        attribute steeringAngle : Real;
    }

    part def Wheel {
        attribute name : String;
    }

    part vehicle1 : Vehicle {
        mass = 1750.0;
        frontAxleAssembly = {
            axle = {
                isSteerable = true,
                steeringAngle = 0.0
            },
            wheel1 = {
                name = "FrontLeft"
            },
            wheel2 = {
                name = "FrontRight"
            }
        };
        rearAxleAssembly = {
            axle = {},
            wheel1 = {
                name = "RearLeft"
            },
            wheel2 = {
                name = "RearRight"
            }
        };
    }

    part vehicle2 : Vehicle {
        mass = 2000.0;
        frontAxleAssembly = {
            axle = {},
            wheel1 = {},
            wheel2 = {}
        };
        rearAxleAssembly = {
            axle = {},
            wheel1 = {},
            wheel2 = {}
        };
    }
}