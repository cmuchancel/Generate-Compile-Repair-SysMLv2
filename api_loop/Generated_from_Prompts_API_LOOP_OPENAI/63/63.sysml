package VehicleMassBudget {
    private import ScalarValues::*;

    enum def MassUnit {
        kg;
    }

    part def Chassis {
        attribute mass : Real;
        attribute massUnit : MassUnit;
    }

    part def Engine {
        attribute mass : Real;
        attribute massUnit : MassUnit;
    }

    part def Transmission {
        attribute mass : Real;
        attribute massUnit : MassUnit;
    }

    part def OtherComponent {
        attribute mass : Real;
        attribute massUnit : MassUnit;
    }

    part def Vehicle {
        attribute massLimit : Real = 2500;
        part chassis : Chassis;
        part engine : Engine;
        part transmission : Transmission;
        part other : OtherComponent;
        attribute totalMass : Real = chassis.mass + engine.mass + transmission.mass + other.mass;
        attribute withinLimit : Boolean = totalMass <= massLimit;
    }
}