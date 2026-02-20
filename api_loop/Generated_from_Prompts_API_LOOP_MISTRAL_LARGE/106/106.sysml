package VehiclePowertrainManagement {
    private import ScalarValues::*;

    enum def EngineSignal {
        enum START;
        enum SHUTDOWN;
    }

    part def Engine {
        attribute fuelCommand : Real;
        attribute outputTorque : Real;
        attribute engineSignal : EngineSignal;
    }

    part def Transmission {
        attribute amplifiedTorque : Real;
    }

    part def Wheel {
        attribute distributedTorque : Real;
    }

    part def PowertrainSystem {
        part engine : Engine;
        part transmission : Transmission;
        part frontLeftWheel : Wheel;
        part frontRightWheel : Wheel;
        part rearLeftWheel : Wheel;
        part rearRightWheel : Wheel;

        connect engine to transmission;
        connect transmission to frontLeftWheel;
        connect transmission to frontRightWheel;
        connect transmission to rearLeftWheel;
        connect transmission to rearRightWheel;
    }

    item def Torque;
}