package VehicleFuelSystem {
    private import ScalarValues::*;

    enum def FuelFlowDirection {
        enum supply;
        enum returnFlow;
    }

    part def FuelTankAssembly {
        attribute capacity : Real;
        attribute currentFuelLevel : Real;
        attribute fuelTemperature : Real;
        port fuelSupplyPort : FuelPort;
        port fuelReturnPort : FuelPort;
    }

    part def Engine {
        attribute fuelConsumptionRate : Real;
        port fuelSupplyPort : FuelPort;
        port fuelReturnPort : FuelPort;
    }

    part def FuelSystem {
        part tank : FuelTankAssembly;
        part engine : Engine;
        attribute systemTemperature : Real;
    }

    port def FuelPort {
        attribute direction : FuelFlowDirection;
    }

    item def Fuel {
        attribute temperature : Real;
    }

    part system : FuelSystem;

    connect system.tank.fuelSupplyPort to system.engine.fuelSupplyPort;
    connect system.engine.fuelReturnPort to system.tank.fuelReturnPort;
}