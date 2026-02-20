package VehicleDynamicsSimulation {
    private import ScalarValues::*;

    port def InputPort {
        attribute initialVelocity : Real;
        attribute initialPosition : Real;
        attribute acceleration : Real;
        attribute timeStep : Real;
    }

    port def OutputPort {
        attribute outputVelocity : Real;
        attribute outputPosition : Real;
    }

    part def ModelCenterTool {
        port inPort : InputPort;
        port outPort : OutputPort;
    }

    part def SimulationSystem {
        port inPort : InputPort;
        port outPort : OutputPort;
        part modelCenter : ModelCenterTool;
        connect inPort to modelCenter.inPort;
        connect modelCenter.outPort to outPort;
    }

    part vehicleDynamicsSimulation : SimulationSystem;
}