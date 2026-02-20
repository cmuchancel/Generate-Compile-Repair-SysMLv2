package VehicleMassAndDynamics {
    private import ScalarValues::*;

    part def Vehicle {
        attribute mass : Real;
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

    part def DynamicBehavior {
        attribute initialVelocity : Real;
        attribute finalVelocity : Real;
        attribute mass : Real;
        attribute force : Real;
        attribute time : Real;
    }

    requirement def TotalMassConstraint {
        attribute maxMass : Real = 2500.0;
    }

    requirement def MassValidity {
        attribute minMass : Real = 0.0;
    }

    requirement def DynamicConstraint {
    }

    part vehicle : Vehicle;
    part dynamic : DynamicBehavior;
    attribute maxMass : Real = 2500.0;

    constraint totalMass {
        vehicle.mass == vehicle.chassis.mass + vehicle.engine.mass + vehicle.transmission.mass;
        vehicle.mass <= maxMass;
    }

    constraint massValidity {
        vehicle.chassis.mass > 0.0;
        vehicle.engine.mass > 0.0;
        vehicle.transmission.mass > 0.0;
        dynamic.mass > 0.0;
    }

    constraint dynamicConstraint {
        dynamic.finalVelocity == dynamic.initialVelocity + (dynamic.force * dynamic.time) / dynamic.mass;
    }
}