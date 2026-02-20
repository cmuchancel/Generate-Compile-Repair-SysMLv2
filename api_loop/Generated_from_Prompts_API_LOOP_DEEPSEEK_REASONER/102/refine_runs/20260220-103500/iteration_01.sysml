package VehiclePowerTransmission {
    private import ScalarValues::*;

    port def ControlPort {
        attribute command : Real;
    }

    port def MechanicalPowerPort {
        attribute torque : Real;
        attribute speed : Real;
    }

    port def MechanicalInterface {
        attribute force : Real;
    }

    part def Engine {
        port fuelIn : ControlPort;
        port powerOut : MechanicalPowerPort;
        port clutchInterface : MechanicalPowerPort;
    }

    part def Transmission {
        port engineIn : MechanicalPowerPort;
        port clutchInterface : MechanicalPowerPort;
        port driveOut : MechanicalPowerPort;
    }

    part def Clutch {
        port engineSide : MechanicalPowerPort;
        port transmissionSide : MechanicalPowerPort;
    }

    part def Driveshaft {
        port transmissionIn : MechanicalPowerPort;
        port axleOut : MechanicalPowerPort;
    }

    part def Differential {
        port driveshaftIn : MechanicalPowerPort;
        port leftOut : MechanicalPowerPort;
        port rightOut : MechanicalPowerPort;
    }

    part def HalfShaft {
        port differentialIn : MechanicalPowerPort;
        port wheelOut : MechanicalPowerPort;
    }

    part def Wheel {
        port axleIn : MechanicalPowerPort;
        port roadInterface : MechanicalInterface;
    }

    part def RearAxleAssembly {
        part differential : Differential;
        part leftHalfShaft : HalfShaft;
        part rightHalfShaft : HalfShaft;

        port driveshaftIn : MechanicalPowerPort;
        port leftWheelOut : MechanicalPowerPort;
        port rightWheelOut : MechanicalPowerPort;

        connect driveshaftIn to differential.driveshaftIn;
        connect differential.leftOut to leftHalfShaft.differentialIn;
        connect differential.rightOut to rightHalfShaft.differentialIn;
        connect leftHalfShaft.wheelOut to leftWheelOut;
        connect rightHalfShaft.wheelOut to rightWheelOut;
    }

    part def VehiclePowerTransmissionSystem {
        part engine : Engine;
        part transmission : Transmission;
        part clutch : Clutch;
        part driveshaft : Driveshaft;
        part rearAxle : RearAxleAssembly;
        part leftRearWheel : Wheel;
        part rightRearWheel : Wheel;

        port fuelCommand : ControlPort;

        connect fuelCommand to engine.fuelIn;
        connect engine.powerOut to clutch.engineSide;
        connect engine.clutchInterface to transmission.clutchInterface;
        connect clutch.transmissionSide to transmission.engineIn;
        connect transmission.driveOut to driveshaft.transmissionIn;
        connect driveshaft.axleOut to rearAxle.driveshaftIn;
        connect rearAxle.leftWheelOut to leftRearWheel.axleIn;
        connect rearAxle.rightWheelOut to rightRearWheel.axleIn;
    }
}