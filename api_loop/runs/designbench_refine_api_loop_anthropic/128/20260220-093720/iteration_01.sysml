package PowerSystem {
    private import ScalarValues::*;

    port def PowerPort;
    port def TorquePort;

    part def TorqueGenerator {
        port torqueOut : TorquePort;
        port powerIn : PowerPort;
    }

    part def Engine {
        port torqueOut : TorquePort;
        port powerIn : PowerPort;
    }

    part def Powertrain {
        part engine : Engine {
            port torqueOut : TorquePort;
            port powerIn : PowerPort;
        }
        port torqueOut : TorquePort;
        port powerIn : PowerPort;
        connect engine.torqueOut to torqueOut;
    }

    part def PowerSystem {
        part torqueGenerator : TorqueGenerator {
            port torqueOut : TorquePort;
            port powerIn : PowerPort;
        }
        part powertrain : Powertrain {
            part engine : Engine {
                port torqueOut : TorquePort;
                port powerIn : PowerPort;
            }
            port torqueOut : TorquePort;
            port powerIn : PowerPort;
        }
        port powerOut : PowerPort;
        port torqueOut : TorquePort;
        connect torqueGenerator.torqueOut to powertrain.torqueOut;
        connect torqueGenerator.torqueOut to torqueOut;
        connect powertrain.engine.torqueOut to torqueGenerator.torqueOut;
    }
}