package VehicleMassManagement {
    private import ScalarValues::*;

    enum def MassStatus {
        enum compliant;
        enum overweight;
    }

    part def Chassis {
        attribute mass : Real;
    }

    part def Engine {
        attribute mass : Real;
    }

    part def Gearbox {
        attribute mass : Real;
    }

    part def Vehicle {
        part chassis : Chassis;
        part engine : Engine;
        part gearbox : Gearbox;
        attribute totalMass : Real;
        attribute status : MassStatus = MassStatus::compliant;
    }

    item def MassCalculator {
        attribute maxPermittedMass : Real = 2500.0;
        attribute calculateTotalMass : Real;
        attribute checkMassStatus : MassStatus;
    }

    part vehicle : Vehicle;
    part massCalculator : MassCalculator;

    attribute chassisMass : Real;
    attribute engineMass : Real;
    attribute gearboxMass : Real;

    vehicle.totalMass = chassisMass + engineMass + gearboxMass;
    vehicle.status = if (vehicle.totalMass <= massCalculator.maxPermittedMass) then MassStatus::compliant else MassStatus::overweight;
}