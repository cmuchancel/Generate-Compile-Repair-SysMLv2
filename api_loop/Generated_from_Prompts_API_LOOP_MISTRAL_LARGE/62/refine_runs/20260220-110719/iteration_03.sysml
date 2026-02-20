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
        attribute calculate(chassisMass : Real, engineMass : Real, gearboxMass : Real) : Real {
            chassisMass + engineMass + gearboxMass
        };
        attribute checkStatus(totalMass : Real) : MassStatus {
            if (totalMass <= maxPermittedMass) then MassStatus::compliant else MassStatus::overweight
        };
    }

    part vehicle : Vehicle;
    part massCalculator : MassCalculator;

    connect vehicle.chassis.mass to massCalculator::calculate.chassisMass;
    connect vehicle.engine.mass to massCalculator::calculate.engineMass;
    connect vehicle.gearbox.mass to massCalculator::calculate.gearboxMass;

    vehicle.totalMass = massCalculator.calculate(vehicle.chassis.mass, vehicle.engine.mass, vehicle.gearbox.mass);
    vehicle.status = massCalculator.checkStatus(vehicle.totalMass);
}