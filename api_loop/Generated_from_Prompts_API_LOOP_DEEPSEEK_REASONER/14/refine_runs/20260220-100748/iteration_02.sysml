package FuelSystem {
    private import ScalarValues::*;

    item def Fuel;
    item def TemperatureInfo;

    port def FuelSupplyPort {
        attribute fuel : Fuel;
    }

    port def FuelReturnPort {
        attribute fuel : Fuel;
    }

    port def TemperaturePort {
        attribute info : TemperatureInfo;
    }

    part def FuelTank {
        port supplyOutlet : FuelSupplyPort;
        port returnInlet : FuelReturnPort;
        port temperature : TemperaturePort;
    }

    part def FuelPump {
        port supplyInlet : FuelSupplyPort;
        port supplyOutlet : FuelSupplyPort;
        port returnInlet : FuelReturnPort;
        port returnOutlet : FuelReturnPort;
    }

    part def FuelTankAssembly {
        port mainSupplyOutlet : FuelSupplyPort;
        port mainReturnInlet : FuelReturnPort;
        port tempSensor : TemperaturePort;

        part tank : FuelTank;
        part pump : FuelPump;

        connect tank.supplyOutlet to pump.supplyInlet;
        connect pump.supplyOutlet to mainSupplyOutlet;
        connect mainReturnInlet to pump.returnInlet;
        connect pump.returnOutlet to tank.returnInlet;
        connect tank.temperature to tempSensor;
    }
}