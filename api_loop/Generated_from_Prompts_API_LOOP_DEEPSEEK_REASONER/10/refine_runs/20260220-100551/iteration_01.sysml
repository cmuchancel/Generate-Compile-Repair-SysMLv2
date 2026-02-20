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
        port fuelOutlet : out FuelSupplyPort;
        port fuelInlet : in FuelReturnPort;
    }

    part def Engine {
        port fuelInlet : in FuelSupplyPort;
        port fuelOutlet : out FuelReturnPort;
    }

    part def FuelFlowSystem {
        part fuelTank : FuelTankAssembly;
        part engine : Engine;
        
        connect fuelTank.fuelOutlet to engine.fuelInlet;
        connect engine.fuelOutlet to fuelTank.fuelInlet;
    }
}