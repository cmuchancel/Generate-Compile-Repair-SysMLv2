package VehiclePowertrainSystem {
    private import ScalarValues::*;

    enum def EngineSignal {
        enum START;
        enum SHUTDOWN;
    }

    item def FuelCommand {
        attribute fuelLevel : Real;
    }

    item def TorqueSignal {
        attribute torqueValue : Real;
    }

    item def PowerSignal {
        attribute powerValue : Real;
    }

    port def FuelInputPort {
        attribute fuelLevel : Real;
    }

    port def TorqueOutputPort {
        attribute torqueValue : Real;
    }

    port def TorqueInputPort {
        attribute torqueValue : Real;
    }

    port def PowerOutputPort {
        attribute powerValue : Real;
    }

    port def PowerInputPort {
        attribute powerValue : Real;
    }

    port def WheelPowerPort {
        attribute powerValue : Real;
    }

    part def Driver {
        attribute fuelCommand : Real;
        attribute engineSignal : EngineSignal;
        port fuelOut : FuelInputPort;
    }

    part def Engine {
        attribute outputTorque : Real;
        attribute engineState : EngineSignal;
        port fuelIn : FuelInputPort;
        port torqueOut : TorqueOutputPort;
    }

    part def Transmission {
        attribute amplificationRatio : Real;
        attribute transmittedTorque : Real;
        port torqueIn : TorqueInputPort;
        port torqueOut : TorqueOutputPort;
    }

    part def TorqueDistributor {
        attribute distributedPower : Real;
        port torqueIn : TorqueInputPort;
        port powerOutFL : PowerOutputPort;
        port powerOutFR : PowerOutputPort;
        port powerOutRL : PowerOutputPort;
        port powerOutRR : PowerOutputPort;
    }

    part def Wheel {
        attribute wheelId : String;
        attribute receivedPower : Real;
        port powerIn : PowerInputPort;
    }

    part def PowertrainControlSystem {
        attribute systemActive : Boolean;
        attribute currentFuelCommand : Real;
        attribute currentEngineSignal : EngineSignal;

        part driver : Driver;
        part engine : Engine;
        part transmission : Transmission;
        part distributor : TorqueDistributor;
        part wheelFL : Wheel;
        part wheelFR : Wheel;
        part wheelRL : Wheel;
        part wheelRR : Wheel;

        connect driver.fuelOut to engine.fuelIn;
        connect engine.torqueOut to transmission.torqueIn;
        connect transmission.torqueOut to distributor.torqueIn;
        connect distributor.powerOutFL to wheelFL.powerIn;
        connect distributor.powerOutFR to wheelFR.powerIn;
        connect distributor.powerOutRL to wheelRL.powerIn;
        connect distributor.powerOutRR to wheelRR.powerIn;
    }
}