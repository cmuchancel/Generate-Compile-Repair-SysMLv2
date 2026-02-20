package FuelSystem {
    private import ScalarValues::*;

    port def FuelPort {
        attribute fuelTemperature : Real;
    }

    part def FuelTank {
        port supply : FuelPort;
        port returnPort : FuelPort;
    }

    part def Engine {
        port fuelInlet : FuelPort;
        port fuelReturn : FuelPort;
    }

    part def VehicleFuelManagement {
        part fuelTank : FuelTank;
        part engine : Engine;

        connect fuelTank.supply to engine.fuelInlet;
        connect engine.fuelReturn to fuelTank.returnPort;
    }
}