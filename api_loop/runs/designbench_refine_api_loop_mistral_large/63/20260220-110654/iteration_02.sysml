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

    part vehicle : Vehicle {
        attribute chassisMass : Real = chassis.mass;
        attribute engineMass : Real = engine.mass;
        attribute transmissionMass : Real = transmission.mass;
        attribute totalMass : Real = chassisMass + engineMass + transmissionMass;
    }

    constraint massConstraint {
        vehicle.totalMass <= vehicle.massLimit
    }
}