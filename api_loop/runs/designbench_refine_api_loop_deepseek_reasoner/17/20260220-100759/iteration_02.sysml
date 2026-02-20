package VehicleFuelSystem {
    private import ScalarValues::*;

    item def Fuel {
        attribute temperature : Real;
    }

    port def FuelInterface {
        attribute currentTemperature : Real;
    }

    part def FuelTankAssembly {
        port supplyPort : FuelInterface;
        port returnPort : FuelInterface;
    }

    part def Engine {
        port fuelSupply : FuelInterface;
        port fuelReturn : FuelInterface;
    }

    part def FuelSystem {
        part tank : FuelTankAssembly;
        part engine : Engine;

        connect tank.supplyPort to engine.fuelSupply;
        connect engine.fuelReturn to tank.returnPort;
    }
}