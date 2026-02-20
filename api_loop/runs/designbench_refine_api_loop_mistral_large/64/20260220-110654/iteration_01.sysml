package VehicleMassCompliance {
    private import ScalarValues::*;

    part def Chassis {
        attribute mass : Real;
    };

    part def Engine {
        attribute mass : Real;
    };

    part def Transmission {
        attribute mass : Real;
    };

    part def Vehicle {
        part chassis : Chassis;
        part engine : Engine;
        part transmission : Transmission;
        attribute totalMass : Real;
        attribute isCompliant : Boolean;
    };

    item def MassData {
        attribute chassisMass : Real;
        attribute engineMass : Real;
        attribute transmissionMass : Real;
    };

    part def MassRegulator {
        port massInput : MassData;
        part vehicle : Vehicle;

        attribute maxAllowedMass : Real = 2500.0;

        connect massInput to vehicle {
            vehicle.chassis.mass = massInput.chassisMass;
            vehicle.engine.mass = massInput.engineMass;
            vehicle.transmission.mass = massInput.transmissionMass;
            vehicle.totalMass = massInput.chassisMass + massInput.engineMass + massInput.transmissionMass;
            vehicle.isCompliant = vehicle.totalMass <= maxAllowedMass;
        };
    };
}