package FuelFlowManagementSystem {
    private import ScalarValues::*;

    item def FuelData {
        attribute flowRate : Real;
        attribute temperature : Real;
    }

    port def FuelSupplyPort {
        attribute fuelData : FuelData;
    }

    port def FuelReturnPort {
        attribute fuelData : FuelData;
    }

    part def FuelTankAssembly {
        port fuelOutlet : FuelSupplyPort;
        port fuelInlet : FuelReturnPort;
    }

    part def Engine {
        port fuelInlet : FuelSupplyPort;
        port fuelOutlet : FuelReturnPort;
    }

    part def FuelFlowSystem {
        part fuelTank : FuelTankAssembly;
        part engine : Engine;

        connect fuelTank.fuelOutlet to engine.fuelInlet;
        connect engine.fuelOutlet to fuelTank.fuelInlet;
    }
}