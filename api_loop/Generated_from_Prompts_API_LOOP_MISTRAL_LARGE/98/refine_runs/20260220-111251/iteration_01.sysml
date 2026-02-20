package FuelManagementSystem {
    private import ScalarValues::*;

    enum def FuelFlowDirection {
        enum supply;
        enum return;
    }

    part def FuelTank {
        attribute capacity : Real;
        port fuelSupplyPort : FuelInterface;
        port fuelReturnPort : FuelInterface;
    }

    part def Engine {
        port fuelSupplyPort : FuelInterface;
        port fuelReturnPort : FuelInterface;
        attribute fuelTemperature : Real;
    }

    port def FuelInterface {
        attribute direction : FuelFlowDirection;
    }

    part def FuelSystem {
        part tank : FuelTank;
        part engine : Engine;

        connect tank.fuelSupplyPort to engine.fuelSupplyPort;
        connect engine.fuelReturnPort to tank.fuelReturnPort;
    }

    item def Fuel {
        attribute temperature : Real;
    }
}