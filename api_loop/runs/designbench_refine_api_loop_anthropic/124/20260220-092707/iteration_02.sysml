package VehicleSystem {
    private import ScalarValues::*;

    part def Wheel {
        attribute mass : Real;
    }

    part def FrontAxle {
        attribute mass : Real;
        attribute steeringAngle : Real;
    }

    part def RearAxle {
        attribute mass : Real;
    }

    part def FrontAxleAssembly {
        attribute mass : Real = 150.0;
        part frontAxle : FrontAxle;
        part frontWheel1 : Wheel;
        part frontWheel2 : Wheel;
    }

    part def RearAxleAssembly {
        attribute mass : Real = 250.0;
        part rearAxle : RearAxle;
        part rearWheel1 : Wheel;
        part rearWheel2 : Wheel;
    }

    part def Car {
        attribute mass : Real = 2500.0;
        part frontAxleAssembly : FrontAxleAssembly;
        part rearAxleAssembly : RearAxleAssembly;
    }

    part car : Car {
        part redefines frontAxleAssembly {
            part redefines frontAxle {
                attribute redefines mass = 50.0;
                attribute redefines steeringAngle = 0.0;
            }
            part redefines frontWheel1 {
                attribute redefines mass = 20.0;
            }
            part redefines frontWheel2 {
                attribute redefines mass = 20.0;
            }
        }
        part redefines rearAxleAssembly {
            part redefines rearAxle {
                attribute redefines mass = 80.0;
            }
            part redefines rearWheel1 {
                attribute redefines mass = 20.0;
            }
            part redefines rearWheel2 {
                attribute redefines mass = 20.0;
            }
        }
    }
}