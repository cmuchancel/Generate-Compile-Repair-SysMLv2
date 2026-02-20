package VehicleSystem {
    private import ScalarValues::*;

    part def Chassis {
        attribute mass : Real;
    }

    part def Engine {
        attribute mass : Real;
    }

    part def Transmission {
        attribute mass : Real;
    }

    part def Vehicle {
        part chassis : Chassis;
        part engine : Engine;
        part transmission : Transmission;

        attribute totalMass : Real = chassis.mass + engine.mass + transmission.mass;
        attribute maxTotalMass : Real = 2500;
        attribute massOk : Boolean = totalMass <= maxTotalMass;
    }

    part def DynamicsModel {
        attribute initialVelocity : Real;
        attribute finalVelocity : Real;
        attribute mass : Real;
        attribute force : Real;
        attribute time : Real;

        attribute deltaVelocity : Real = finalVelocity - initialVelocity;
        attribute massPositive : Boolean = mass > 0;
        attribute forceRelationOk : Boolean = force == mass * deltaVelocity / time;
    }

    part def VehicleSystemModel {
        part vehicle : Vehicle;
        part dynamics : DynamicsModel;
    }
}