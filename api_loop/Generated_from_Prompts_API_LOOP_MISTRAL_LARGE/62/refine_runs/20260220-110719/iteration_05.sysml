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
        attribute calculate(chassisMass : Real, engineMass : Real, gearboxMass : Real) : Real = chassisMass + engineMass + gearboxMass;
        attribute checkStatus(totalMass : Real) : MassStatus = if (totalMass <= maxPermittedMass) then MassStatus::compliant else MassStatus::overweight;
    }

    part vehicle : Vehicle;
    part massCalculator : MassCalculator;

    attribute totalMassCalc : Real = massCalculator.calculate(vehicle.chassis.mass, vehicle.engine.mass, vehicle.gearbox.mass);
    attribute statusCheck : MassStatus = massCalculator.checkStatus(totalMassCalc);

    vehicle.totalMass = totalMassCalc;
    vehicle.status := statusCheck;
}