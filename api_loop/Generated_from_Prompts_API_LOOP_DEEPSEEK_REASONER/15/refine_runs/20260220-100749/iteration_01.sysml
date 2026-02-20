package FuelSystem {
    private import ScalarValues::*;

    item def Fuel {
        attribute temperature : Real;
    }

    port def FuelSupplyPort {
        attribute flow : Fuel;
    }

    port def FuelReturnPort {
        attribute flow : Fuel;
    }

    port def TemperatureDataPort {
        attribute data : Real;
    }

    part def FuelTankAssembly {
        port supplyOutlet : FuelSupplyPort;
        port returnInlet : FuelReturnPort;
        port tempSensor : TemperatureDataPort;
    }

    part def Engine {
        port supplyInlet : FuelSupplyPort;
        port returnOutlet : FuelReturnPort;
    }

    part def Vehicle {
        part fuelTank : FuelTankAssembly;
        part engine : Engine;
        part supplyLine : FuelLine;
        part returnLine : FuelLine;

        connect fuelTank.supplyOutlet to supplyLine.inlet;
        connect supplyLine.outlet to engine.supplyInlet;
        connect engine.returnOutlet to returnLine.inlet;
        connect returnLine.outlet to fuelTank.returnInlet;
        connect fuelTank.tempSensor to vehicleMonitor.tempInput;
    }

    part def FuelLine {
        port inlet : FuelSupplyPort;
        port outlet : FuelSupplyPort;
    }

    part def VehicleMonitor {
        port tempInput : TemperatureDataPort;
    }
}