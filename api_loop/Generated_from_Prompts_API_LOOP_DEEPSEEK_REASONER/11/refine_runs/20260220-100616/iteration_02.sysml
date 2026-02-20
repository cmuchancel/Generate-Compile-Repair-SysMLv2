package FuelSystem {
    private import ScalarValues::*;

    item def Fuel {
        attribute temperature : Real;
    }

    port def FuelSupplyPort;
    port def FuelReturnPort;

    part def FuelTankAssembly {
        port fuelSupply : FuelSupplyPort;
        port fuelReturn : FuelReturnPort;
        attribute fuelLevel : Real;
    }

    part def Engine {
        port fuelIntake : FuelSupplyPort;
        port fuelOutlet : FuelReturnPort;
    }

    part def MonitoringSystem {
        attribute fuelTemperature : Real;
    }

    part def VehicleFuelSystem {
        part tank : FuelTankAssembly;
        part engine : Engine;
        part monitor : MonitoringSystem;

        connect tank.fuelSupply to engine.fuelIntake;
        connect engine.fuelOutlet to tank.fuelReturn;
    }
}