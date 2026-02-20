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
        part chassis : Chassis {
            attribute :>> mass = 800.0;
        }
        part engine : Engine {
            attribute :>> mass = 500.0;
        }
        part transmission : Transmission {
            attribute :>> mass = 200.0;
        }
        attribute totalMass : Real = 800.0 + 500.0 + 200.0;
        attribute maxMass : Real = 2500.0;
        attribute massConstraintSatisfied : Boolean = (1500.0 <= 2500.0);
    }

    part def VehicleDynamics {
        attribute initialVelocity : Real;
        attribute finalVelocity : Real;
        attribute mass : Real;
        attribute force : Real;
        attribute time : Real;
        attribute deltaVelocity : Real = 10.0;
        attribute massPositive : Boolean = (1500.0 > 0.0);
        attribute impulse : Real = 500.0 * 3.0;
        attribute momentumChange : Real = 1500.0 * 10.0;
        attribute dynamicConstraintSatisfied : Boolean = (1500.0 > 0.0);
    }

    part vehicle : Vehicle;

    part dynamics : VehicleDynamics {
        attribute :>> initialVelocity = 0.0;
        attribute :>> finalVelocity = 10.0;
        attribute :>> mass = 1500.0;
        attribute :>> force = 500.0;
        attribute :>> time = 30.0;
    }
}