package VehicleMassModel {
    private import ScalarValues::*;

    part def Vehicle;
    part def Engine;
    part def Transmission;
    part def Body;
    part def Component;

    part def Vehicle {
        attribute baseMass : Real;
        attribute totalMass : Real;

        part engine : Engine;
        part transmission : Transmission;
        part body : Body;
        part otherComponents : Component[0..*];
    }

    part def Engine {
        attribute mass : Real;
    }

    part def Transmission {
        attribute mass : Real;
    }

    part def Body {
        attribute mass : Real;
    }

    part def Component {
        attribute mass : Real;
        attribute totalMass : Real;
    }

    part vehicle : Vehicle;

    part vehicle_example : Vehicle {
        baseMass = 1000.0;
        part engine : Engine {
            mass = 200.0;
        }
        part transmission : Transmission {
            mass = 150.0;
        }
        part body : Body {
            mass = 500.0;
        }
        part otherComponents : Component[3] {
            part comp1 : Component {
                mass = 50.0;
            }
            part comp2 : Component {
                mass = 30.0;
            }
            part comp3 : Component {
                mass = 20.0;
            }
        }
    }
}