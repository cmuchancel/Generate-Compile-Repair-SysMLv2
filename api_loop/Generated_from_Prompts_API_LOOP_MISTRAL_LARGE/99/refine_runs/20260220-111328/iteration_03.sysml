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
            part axle : FrontAxle;
            part wheel1 : Wheel;
            part wheel2 : Wheel;
        };
        rearAxleAssembly = {
            part axle : Axle;
            part wheel1 : Wheel;
            part wheel2 : Wheel;
        };
    }

    part vehicle2 : Vehicle {
        mass = 2000.0;
        frontAxleAssembly = {
            part axle : FrontAxle;
            part wheel1 : Wheel;
            part wheel2 : Wheel;
        };
        rearAxleAssembly = {
            part axle : Axle;
            part wheel1 : Wheel;
            part wheel2 : Wheel;
        };
    }

    part vehicle1_frontAxle : FrontAxle {
        :>> vehicle1.frontAxleAssembly.axle;
        steeringAngle = 0.0;
    }

    part vehicle1_leftFrontWheel : Wheel {
        :>> vehicle1.frontAxleAssembly.wheel1;
        name = "Left Front Wheel";
    }

    part vehicle1_rightFrontWheel : Wheel {
        :>> vehicle1.frontAxleAssembly.wheel2;
        name = "Right Front Wheel";
    }

    part vehicle1_leftRearWheel : Wheel {
        :>> vehicle1.rearAxleAssembly.wheel1;
        name = "Left Rear Wheel";
    }

    part vehicle1_rightRearWheel : Wheel {
        :>> vehicle1.rearAxleAssembly.wheel2;
        name = "Right Rear Wheel";
    }

    part vehicle2_frontAxle : FrontAxle {
        :>> vehicle2.frontAxleAssembly.axle;
        steeringAngle = 0.0;
    }
}