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
        attribute redefines mass = 2500.0;
        part redefines frontAxleAssembly : FrontAxleAssembly {
            attribute redefines mass = 150.0;
            part redefines frontAxle : FrontAxle {
                attribute redefines mass = 50.0;
                attribute redefines steeringAngle = 0.0;
            }
            part redefines frontWheel1 : Wheel {
                attribute redefines mass = 20.0;
            }
            part redefines frontWheel2 : Wheel {
                attribute redefines mass = 20.0;
            }
        }
        part redefines rearAxleAssembly : RearAxleAssembly {
            attribute redefines mass = 250.0;
            part redefines rearAxle : RearAxle {
                attribute redefines mass = 80.0;
            }
            part redefines rearWheel1 : Wheel {
                attribute redefines mass = 20.0;
            }
            part redefines rearWheel2 : Wheel {
                attribute redefines mass = 20.0;
            }
        }
    }
}