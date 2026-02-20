package PowerTransmissionSystem {
    private import ScalarValues::*;

    item def Power {
        attribute value : Real;
    }

    item def Torque {
        attribute value : Real;
    }

    port def PowerPort {
        port output : Power;
    }

    port def TorquePort {
        port output : Torque;
    }

    part def Engine {
        port torqueOutput : TorquePort;
        port powerOutput : PowerPort;
        attribute generatedTorque : Real;
        attribute generatedPower : Real;
    }

    part def TorqueGenerator {
        port torqueOut : TorquePort;
        attribute commandedTorque : Real;
    }

    part def Powertrain {
        part engine : Engine;
    }

    part def System {
        part torqueGen : TorqueGenerator;
        part powertrain : Powertrain;
        port systemPowerOut : PowerPort;

        connect torqueGen.torqueOut to powertrain.engine.torqueOutput;
        connect powertrain.engine.powerOutput to systemPowerOut;
    }
}