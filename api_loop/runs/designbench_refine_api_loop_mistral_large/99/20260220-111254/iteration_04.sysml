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
        attribute steeringAngle : Real = 0.0;
    }

    part def Wheel {
        attribute name : String = "";
    }

    part vehicle1 {
        :>> Vehicle;
        mass = 1750.0;
        frontAxleAssembly = {
            :>> AxleAssembly;
            axle = {
                :>> Axle;
                isSteerable = true;
                steeringAngle = 0.0;
            };
            wheel1 = {
                :>> Wheel;
                name = "FrontLeft";
            };
            wheel2 = {
                :>> Wheel;
                name = "FrontRight";
            };
        };
        rearAxleAssembly = {
            :>> AxleAssembly;
            axle = {
                :>> Axle;
            };
            wheel1 = {
                :>> Wheel;
                name = "RearLeft";
            };
            wheel2 = {
                :>> Wheel;
                name = "RearRight";
            };
        };
    }

    part vehicle2 {
        :>> Vehicle;
        mass = 2000.0;
        frontAxleAssembly = {
            :>> AxleAssembly;
            axle = {
                :>> Axle;
            };
            wheel1 = {
                :>> Wheel;
            };
            wheel2 = {
                :>> Wheel;
            };
        };
        rearAxleAssembly = {
            :>> AxleAssembly;
            axle = {
                :>> Axle;
            };
            wheel1 = {
                :>> Wheel;
            };
            wheel2 = {
                :>> Wheel;
            };
        };
    }
}