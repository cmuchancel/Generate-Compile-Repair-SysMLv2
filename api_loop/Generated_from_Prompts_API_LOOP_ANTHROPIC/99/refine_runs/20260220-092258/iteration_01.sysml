package VehicleStructure {
    private import ScalarValues::*;

    part def Wheel {
        attribute wheelName : String;
    }

    part def Axle {
        part wheel1 : Wheel;
        part wheel2 : Wheel;
    }

    part def SteeringAxle :> Axle {
        attribute steeringAngle : Real;
    }

    part def AxleAssembly {
        part axle : Axle;
        part wheel1 : Wheel;
        part wheel2 : Wheel;
    }

    part def Vehicle {
        attribute mass : Real;
        part frontAxleAssembly : AxleAssembly;
        part rearAxleAssembly : AxleAssembly;
    }

    part vehicle1 : Vehicle {
        attribute redefines mass = 1750;
        part redefines frontAxleAssembly : AxleAssembly {
            part redefines axle : SteeringAxle {
                attribute redefines steeringAngle = 0.0;
                part redefines wheel1 : Wheel {
                    attribute redefines wheelName = "FrontLeft";
                }
                part redefines wheel2 : Wheel {
                    attribute redefines wheelName = "FrontRight";
                }
            }
            part redefines wheel1 : Wheel {
                attribute redefines wheelName = "FrontLeft";
            }
            part redefines wheel2 : Wheel {
                attribute redefines wheelName = "FrontRight";
            }
        }
        part redefines rearAxleAssembly : AxleAssembly {
            part redefines axle : Axle {
                part redefines wheel1 : Wheel {
                    attribute redefines wheelName = "RearLeft";
                }
                part redefines wheel2 : Wheel {
                    attribute redefines wheelName = "RearRight";
                }
            }
            part redefines wheel1 : Wheel {
                attribute redefines wheelName = "RearLeft";
            }
            part redefines wheel2 : Wheel {
                attribute redefines wheelName = "RearRight";
            }
        }
    }

    part vehicle2 : Vehicle {
        attribute redefines mass = 2000;
        part redefines frontAxleAssembly : AxleAssembly {
            part redefines axle : SteeringAxle {
                attribute redefines steeringAngle = 0.0;
            }
        }
    }
}