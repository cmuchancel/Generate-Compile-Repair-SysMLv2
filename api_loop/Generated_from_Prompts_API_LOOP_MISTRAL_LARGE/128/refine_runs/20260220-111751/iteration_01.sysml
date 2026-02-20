package TorqueSystem {
    private import ScalarValues::*;

    part def TorqueGenerator {
        attribute torqueOutput : Real;
        port torqueOut : TorquePort;
    }

    part def Engine {
        attribute torqueOutput : Real;
        port torqueOut : TorquePort;
    }

    part def Powertrain {
        part engine : Engine;
    }

    part def System {
        part torqueGenerator : TorqueGenerator;
        part powertrain : Powertrain;

        connect torqueGenerator.torqueOut to powertrain.engine.torqueOut;
    }

    port def TorquePort;
}