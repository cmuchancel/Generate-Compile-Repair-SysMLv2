package FuelSystem {
    private import ScalarValues::*;

    port def FuelFlowPort;
    port def TemperatureDataPort;

    part def FuelTank {
        attribute capacity : Real;
        attribute currentLevel : Real;
        port outlet : FuelFlowPort;
        port returnInlet : FuelFlowPort;
        port tempSensor : TemperatureDataPort;
    }

    part def FuelPump {
        attribute pressure : Real;
        port inlet : FuelFlowPort;
        port outlet : FuelFlowPort;
        port returnOutlet : FuelFlowPort;
    }

    part def FuelConsumingUnit {
        port fuelInlet : FuelFlowPort;
        port fuelReturnOutlet : FuelFlowPort;
    }

    part def FuelTankAssembly {
        part tank : FuelTank;
        part pump : FuelPump;
        port supplyOutlet : FuelFlowPort;
        port returnInlet : FuelFlowPort;
        port temperatureOutput : TemperatureDataPort;

        connect pump.outlet to supplyOutlet;
        connect pump.inlet to tank.outlet;
        connect returnInlet to tank.returnInlet;
        connect pump.returnOutlet to tank.returnInlet;
        connect tank.tempSensor to temperatureOutput;
    }

    part def VehicleFuelSystem {
        part assembly : FuelTankAssembly;
        part consumer1 : FuelConsumingUnit;
        part consumer2 : FuelConsumingUnit;

        connect assembly.supplyOutlet to consumer1.fuelInlet;
        connect assembly.supplyOutlet to consumer2.fuelInlet;
        connect consumer1.fuelReturnOutlet to assembly.returnInlet;
        connect consumer2.fuelReturnOutlet to assembly.returnInlet;
    }
}