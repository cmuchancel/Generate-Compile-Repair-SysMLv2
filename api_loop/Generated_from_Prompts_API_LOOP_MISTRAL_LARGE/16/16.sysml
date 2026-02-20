package VehicleFuelSystem {
    private import ScalarValues::*;

    enum def FlowDirection {
        enum supply;
        enum returnFlow;
    }

    part def FuelTankAssembly {
        attribute fuelLevel : Real;
        attribute fuelTemperature : Real;
        port fuelOutlet : FuelPort;
        port fuelReturnInlet : FuelPort;
    }

    part def EngineUnit {
        attribute fuelConsumptionRate : Real;
        port fuelInlet : FuelPort;
        port fuelReturnOutlet : FuelPort;
    }

    port def FuelPort {
        attribute fuelTemperature : Real;
        attribute flowDirection : FlowDirection;
    }

    item def Fuel {
        attribute temperature : Real;
    }

    part vehicleFuelSystem {
        part tank : FuelTankAssembly;
        part engine : EngineUnit;

        connect tank.fuelOutlet to engine.fuelInlet;
        connect engine.fuelReturnOutlet to tank.fuelReturnInlet;
    }
}