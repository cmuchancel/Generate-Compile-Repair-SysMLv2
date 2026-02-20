package VehicleProject {
    private import ScalarValues::*;

    port def ClutchInterface;

    item def TorqueSpeedPerformanceTable {
        attribute data : String;
    }

    part def Engine {
        port torqueOutput : ClutchInterface;
        attribute performanceTable : TorqueSpeedPerformanceTable;
    }

    part def Vehicle {
        attribute emptyMass : Mass = 1500kg;
        attribute fullyFueledMass : Mass = 2000kg;
        part engine : Engine;
        port transmissionInput : ClutchInterface;
        connect engine.torqueOutput to transmissionInput;
    }
}