package FuelSystem {
    private import ScalarValues::*;

    item def Fuel {
        attribute temperature : Real;
    }

    port def FuelSupplyPort;
    port def FuelReturnPort;
    port def TemperatureDataPort;

    part def FuelTankAssembly {
        port supplyOutlet : FuelSupplyPort;
        port returnInlet : FuelReturnPort;
        port tempSensor : TemperatureDataPort;
    }

    part def Engine {
        port supplyInlet : FuelSupplyPort;
        port returnOutlet : FuelReturnPort;
    }

    part def FuelSupplyLine {
        port inlet : FuelSupplyPort;
        port outlet : FuelSupplyPort;
    }

    part def FuelReturnLine {
        port inlet : FuelReturnPort;
        port outlet : FuelReturnPort;
    }

    part def VehicleMonitor {
        port tempInput : TemperatureDataPort;
    }

    part def Vehicle {
        part fuelTank : FuelTankAssembly;
        part engine : Engine;
        part supplyLine : FuelSupplyLine;
        part returnLine : FuelReturnLine;
        part monitor : VehicleMonitor;

        connect fuelTank.supplyOutlet to supplyLine.inlet;
        connect supplyLine.outlet to engine.supplyInlet;
        connect engine.returnOutlet to returnLine.inlet;
        connect returnLine.outlet to fuelTank.returnInlet;
        connect fuelTank.tempSensor to monitor.tempInput;
    }
}