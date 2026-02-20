package VehiclePowerTransmissionSystem {
    private import ScalarValues::*;

    enum def ClutchState {
        enum engaged;
        enum disengaged;
    }

    enum def Gear {
        enum neutral;
        enum gear1;
        enum gear2;
        enum gear3;
        enum gear4;
        enum gear5;
        enum reverse;
    }

    port def PowerPort {
        attribute direction : String = "inout";
    }

    port def ControlPort {
        attribute direction : String = "in";
    }

    port def RoadInterfacePort {
        attribute direction : String = "out";
    }

    item def Torque {
        attribute value : Real;
    }

    item def Speed {
        attribute value : Real;
    }

    item def FuelCommand {
        attribute value : Real;
    }

    part def Engine {
        port fuelCommandPort : ControlPort;
        port powerOutPort : PowerPort;
        attribute currentTorque : Torque;
        attribute currentSpeed : Speed;
    }

    part def Clutch {
        port powerInPort : PowerPort;
        port powerOutPort : PowerPort;
        port controlPort : ControlPort;
        attribute clutchState : ClutchState = ClutchState::disengaged;
    }

    part def Transmission {
        port powerInPort : PowerPort;
        port powerOutPort : PowerPort;
        port controlPort : ControlPort;
        attribute currentGear : Gear = Gear::neutral;
    }

    part def Driveshaft {
        port powerInPort : PowerPort;
        port powerOutPort : PowerPort;
    }

    part def Differential {
        port powerInPort : PowerPort;
        port leftPowerOutPort : PowerPort;
        port rightPowerOutPort : PowerPort;
    }

    part def HalfShaft {
        port powerInPort : PowerPort;
        port wheelInterfacePort : PowerPort;
    }

    part def Wheel {
        port axleInterfacePort : PowerPort;
        port roadInterfacePort : RoadInterfacePort;
        attribute radius : Real;
    }

    part def RearAxleAssembly {
        part differential : Differential;
        part leftHalfShaft : HalfShaft;
        part rightHalfShaft : HalfShaft;

        connect differential.leftPowerOutPort to leftHalfShaft.powerInPort;
        connect differential.rightPowerOutPort to rightHalfShaft.powerInPort;
    }

    part vehiclePowerTransmissionSystem {
        part engine : Engine;
        part clutch : Clutch;
        part transmission : Transmission;
        part driveshaft : Driveshaft;
        part rearAxle : RearAxleAssembly;
        part leftRearWheel : Wheel;
        part rightRearWheel : Wheel;

        connect engine.powerOutPort to clutch.powerInPort;
        connect clutch.powerOutPort to transmission.powerInPort;
        connect transmission.powerOutPort to driveshaft.powerInPort;
        connect driveshaft.powerOutPort to rearAxle.differential.powerInPort;
        connect rearAxle.leftHalfShaft.wheelInterfacePort to leftRearWheel.axleInterfacePort;
        connect rearAxle.rightHalfShaft.wheelInterfacePort to rightRearWheel.axleInterfacePort;
    }
}