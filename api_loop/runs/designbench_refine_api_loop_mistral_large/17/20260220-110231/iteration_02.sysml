package FuelSystemManagement {
    private import ScalarValues::*;

    enum def FuelFlowDirection {
        enum supply;
        enum returnFlow;
    }

    part def FuelTankAssembly {
        attribute capacity : Real;
        attribute currentFuelLevel : Real;
        attribute fuelTemperature : Real;

        port fuelSupplyPort : FuelSupplyPort;
        port fuelReturnPort : FuelReturnPort;
    }

    part def Engine {
        attribute fuelConsumptionRate : Real;
        attribute fuelTemperature : Real;

        port fuelSupplyPort : FuelSupplyPort;
        port fuelReturnPort : FuelReturnPort;
    }

    port def FuelSupplyPort {
        attribute fuelTemperature : Real;
    }

    port def FuelReturnPort {
        attribute fuelTemperature : Real;
    }

    item def Fuel {
        attribute temperature : Real;
    }

    part fuelTank : FuelTankAssembly;
    part engine : Engine;

    connect fuelTank.fuelSupplyPort to engine.fuelSupplyPort;
    connect engine.fuelReturnPort to fuelTank.fuelReturnPort;
}