package VehiclePowerTransmissionSystem {
    private import ScalarValues::*;

    enum def PowerFlowDirection {
        in
        out
        inout
    }

    port def PowerPort {
        attribute direction : PowerFlowDirection;
    }

    port def ClutchPort {
        attribute engaged : Boolean;
    }

    port def CommandPort {
        attribute command : String;
    }

    port def RoadInterfacePort {
        attribute frictionCoefficient : Real;
    }

    part def Engine {
        port fuelCommand : CommandPort;
        port powerOut : PowerPort;
        port clutchIn : ClutchPort;
    }

    part def Transmission {
        port powerIn : PowerPort;
        port clutchOut : ClutchPort;
        port powerOut : PowerPort;
        port fuelCommand : CommandPort;
    }

    part def Clutch {
        port engineSide : ClutchPort;
        port transmissionSide : ClutchPort;
    }

    part def Driveshaft {
        port powerIn : PowerPort;
        port powerOut : PowerPort;
    }

    part def Differential {
        port powerIn : PowerPort;
        port leftPowerOut : PowerPort;
        port rightPowerOut : PowerPort;
    }

    part def HalfShaft {
        port powerIn : PowerPort;
        port wheelSide : PowerPort;
    }

    part def Wheel {
        port axleSide : PowerPort;
        port roadInterface : RoadInterfacePort;
    }

    part def RearAxleAssembly {
        part differential : Differential;
        part leftHalfShaft : HalfShaft;
        part rightHalfShaft : HalfShaft;

        connect differential.leftPowerOut to leftHalfShaft.powerIn;
        connect differential.rightPowerOut to rightHalfShaft.powerIn;
    }

    part vehiclePowerTransmissionSystem {
        part engine : Engine;
        part clutch : Clutch;
        part transmission : Transmission;
        part driveshaft : Driveshaft;
        part rearAxle : RearAxleAssembly;
        part leftRearWheel : Wheel;
        part rightRearWheel : Wheel;

        connect engine.powerOut to clutch.engineSide;
        connect clutch.transmissionSide to transmission.clutchOut;
        connect transmission.powerOut to driveshaft.powerIn;
        connect driveshaft.powerOut to rearAxle.differential.powerIn;
        connect rearAxle.leftHalfShaft.wheelSide to leftRearWheel.axleSide;
        connect rearAxle.rightHalfShaft.wheelSide to rightRearWheel.axleSide;
        connect engine.fuelCommand to transmission.fuelCommand;
    }
}