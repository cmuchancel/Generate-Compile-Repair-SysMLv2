package VehicleMassBudget {
    private import ScalarValues::*;

    part def Vehicle {
        attribute chassisMass : Real;
        attribute engineMass : Real;
        attribute transmissionMass : Real;
        attribute totalMass : Real;
        attribute massLimit : Real = 2500.0;

        part chassis : Chassis;
        part engine : Engine;
        part transmission : Transmission;
    }

    part def Chassis {
        attribute mass : Real;
    }

    part def Engine {
        attribute mass : Real;
    }

    part def Transmission {
        attribute mass : Real;
    }

    part vehicle : Vehicle;

    attribute vehicle.chassisMass : Real = vehicle.chassis.mass;
    attribute vehicle.engineMass : Real = vehicle.engine.mass;
    attribute vehicle.transmissionMass : Real = vehicle.transmission.mass;
    attribute vehicle.totalMass : Real = vehicle.chassisMass + vehicle.engineMass + vehicle.transmissionMass;

    constraint massConstraint {
        vehicle.totalMass <= vehicle.massLimit
    }
}