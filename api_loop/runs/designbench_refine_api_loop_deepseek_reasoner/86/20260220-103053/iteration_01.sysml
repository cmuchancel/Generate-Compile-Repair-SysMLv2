package VehicleDynamics {
    private import ScalarValues::*;

    part def Vehicle {
        attribute initialVelocity : Real;
        attribute initialPosition : Real;
        attribute acceleration : Real;
        attribute timeStep : Real;
        attribute calculatedVelocity : Real;
        attribute calculatedPosition : Real;
    }

    part def ModelCenter {
        port analysisInput : AnalysisPort;
        port analysisOutput : ResultPort;
    }

    port def AnalysisPort;
    port def ResultPort;

    part vehicle : Vehicle;
    part modelCenter : ModelCenter;

    connect vehicle to modelCenter.analysisInput;
    connect modelCenter.analysisOutput to vehicle;
}