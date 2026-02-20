package VehicleMassManagement {
    private import ScalarValues::*;

    enum def MassStatus {
        enum compliant
        enum overweight
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
        attribute calculateTotalMass(chassisMass : Real, engineMass : Real, gearboxMass : Real) : Real;
        attribute checkMassStatus(totalMass : Real) : MassStatus;
    }

    part vehicle : Vehicle;
    part massCalculator : MassCalculator;

    attribute vehicle.chassis.mass : Real;
    attribute vehicle.engine.mass : Real;
    attribute vehicle.gearbox.mass : Real;

    vehicle.totalMass = massCalculator.calculateTotalMass(vehicle.chassis.mass, vehicle.engine.mass, vehicle.gearbox.mass);
    vehicle.status = massCalculator.checkMassStatus(vehicle.totalMass);
}