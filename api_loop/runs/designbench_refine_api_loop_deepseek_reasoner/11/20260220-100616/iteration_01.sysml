package FuelSystem {
    private import ScalarValues::*;

    item def Fuel {
        attribute temperature : Real;
    }

    port def FuelSupplyPort;
    port def FuelReturnPort;

    part def FuelTankAssembly {
        port supply : FuelSupplyPort;
        port return : FuelReturnPort;
        attribute fuelLevel : Real;
    }

    part def Engine {
        port intake : FuelSupplyPort;
        port outlet : FuelReturnPort;
    }

    part def MonitoringSystem {
        attribute fuelTemperature : Real;
    }

    part def VehicleFuelSystem {
        part tank : FuelTankAssembly;
        part engine : Engine;
        part monitor : MonitoringSystem;

        connect tank.supply to engine.intake;
        connect engine.outlet to tank.return;
    }
}