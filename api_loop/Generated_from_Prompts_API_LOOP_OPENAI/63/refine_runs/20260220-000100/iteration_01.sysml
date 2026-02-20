package VehicleMassBudget {
    private import ScalarValues::*;

    enum def MassUnit {
        enum kg;
    }

    part def Component {
        attribute mass : Real;
        attribute massUnit : MassUnit = kg;
    }

    part def Engine : Component;
    part def Transmission : Component;
    part def Chassis : Component;
    part def OtherComponent : Component;

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