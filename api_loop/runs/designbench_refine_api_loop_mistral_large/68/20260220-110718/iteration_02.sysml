package VehicleSystem {
    private import ScalarValues::*;

    enum def MassUnit {
        enum kilogram;
        enum gram;
    }

    part def Mass {
        attribute value : Real;
        attribute unit : MassUnit = MassUnit::kilogram;
    }

    part def Engine {
        attribute speed : Real;
        attribute torque : Real;
        port clutchInterface : ClutchPort;
    }

    part def Clutch {
        port input : ClutchPort;
        port output : ClutchPort;
    }

    part def Transmission {
        port input : ClutchPort;
    }

    part def Fuel {
        attribute mass : Mass;
    }

    part def Vehicle {
        attribute curbWeight : Mass;
        attribute maxMassLimit : Mass;
        part fuel : Fuel;
        part engine : Engine;
        part clutch : Clutch;
        part transmission : Transmission;

        attribute totalMassValue : Real = curbWeight.value + fuel.mass.value;
        attribute totalMass : Mass = Mass(totalMassValue, MassUnit::kilogram);

        constraint maxMassConstraint {
            totalMass.value <= maxMassLimit.value
        }

        constraint positiveFuelMass {
            fuel.mass.value > 0.0
        }
    }

    port def ClutchPort;

    part vehicle : Vehicle;

    connect vehicle.engine.clutchInterface to vehicle.clutch.input;
    connect vehicle.clutch.output to vehicle.transmission.input;
}