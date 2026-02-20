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
        attribute emptyMass : Real = 1500;
        attribute fullyFueledMass : Real = 2000;
        part engine : Engine;
        port transmissionInput : ClutchInterface;
        connect engine.torqueOutput to transmissionInput;
    }
}