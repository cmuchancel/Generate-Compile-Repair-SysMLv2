package VehiclePowerTransmission {
    private import ScalarValues::*;

    item def Power;
    item def Command;
    item def RoadForce;

    port def PowerPort {}
    port def CommandPort {}
    port def RoadPort {}

    part def FuelController {
        port commandOut : CommandPort;
    }

    part def Engine {
        port fuelCmd : CommandPort;
        port powerOut : PowerPort;
    }

    part def Clutch {
        port engineIn : PowerPort;
        port transOut : PowerPort;
    }

    part def Transmission {
        port clutchIn : PowerPort;
        port powerOut : PowerPort;
    }

    part def DriveShaft {
        port in : PowerPort;
        port out : PowerPort;
    }

    part def Differential {
        port in : PowerPort;
        port outLeft : PowerPort;
        port outRight : PowerPort;
    }

    part def HalfShaft {
        port in : PowerPort;
        port out : PowerPort;
    }

    part def RearAxleAssembly {
        port driveshaftIn : PowerPort;
        port leftOut : PowerPort;
        port rightOut : PowerPort;

        part diff : Differential;
        part leftHalf : HalfShaft;
        part rightHalf : HalfShaft;

        connect driveshaftIn to diff.in;
        connect diff.outLeft to leftHalf.in;
        connect diff.outRight to rightHalf.in;
        connect leftHalf.out to leftOut;
        connect rightHalf.out to rightOut;
    }

    part def Wheel {
        port axleIn : PowerPort;
        port roadOut : RoadPort;
    }

    part def RoadSurface {
        port leftIn : RoadPort;
        port rightIn : RoadPort;
    }

    part def VehiclePowerTransmissionSystem {
        part controller : FuelController;
        part engine : Engine;
        part clutch : Clutch;
        part transmission : Transmission;
        part driveshaft : DriveShaft;
        part rearAxle : RearAxleAssembly;
        part leftWheel : Wheel;
        part rightWheel : Wheel;
        part road : RoadSurface;

        connect controller.commandOut to engine.fuelCmd;
        connect engine.powerOut to clutch.engineIn;
        connect clutch.transOut to transmission.clutchIn;
        connect transmission.powerOut to driveshaft.in;
        connect driveshaft.out to rearAxle.driveshaftIn;
        connect rearAxle.leftOut to leftWheel.axleIn;
        connect rearAxle.rightOut to rightWheel.axleIn;
        connect leftWheel.roadOut to road.leftIn;
        connect rightWheel.roadOut to road.rightIn;
    }
}