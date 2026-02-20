package FuelSystem {
    private import ScalarValues::*;

    enum def FuelFlowState {
        enum Flowing;
        enum Idle;
        enum Returning;
    }

    item def FuelItem;
    item def ReturnedFuelItem;
    item def TemperatureDataItem;

    port def FuelOutletPort {
        out item fuelOut : FuelItem;
    }

    port def FuelReturnPort {
        in item fuelReturn : ReturnedFuelItem;
    }

    port def FuelInletPort {
        in item fuelIn : FuelItem;
    }

    port def FuelSupplyPort {
        out item fuelSupply : FuelItem;
    }

    port def TemperaturePort {
        out item tempData : TemperatureDataItem;
    }

    part def FuelTank {
        attribute capacity : Real;
        attribute currentLevel : Real;
        attribute fuelTemperature : Real;

        port tankOutlet : FuelOutletPort;
        port tankReturn : FuelReturnPort;
        port tempMonitor : TemperaturePort;
    }

    part def FuelPump {
        attribute pumpFlowRate : Real;
        attribute isOperational : Boolean;
        attribute flowState : FuelFlowState;

        port pumpInlet : FuelInletPort;
        port pumpOutlet : FuelSupplyPort;
        port pumpReturn : FuelReturnPort;
        port pumpTemp : TemperaturePort;
    }

    part def FuelTankAssembly {
        attribute assemblyId : String;
        attribute returnEnabled : Boolean;

        port assemblyOutlet : FuelOutletPort;
        port assemblyReturn : FuelReturnPort;
        port assemblyTemp : TemperaturePort;

        part tank : FuelTank;
        part pump : FuelPump;

        connect tank.tankOutlet to pump.pumpInlet;
        connect tank.tankReturn to pump.pumpReturn;
        connect tank.tempMonitor to pump.pumpTemp;
        connect pump.pumpOutlet to assemblyOutlet;
        connect assemblyReturn to tank.tankReturn;
        connect assemblyTemp to tank.tempMonitor;
    }

    part def Engine {
        attribute engineId : String;
        attribute fuelDemand : Real;

        port engineFuelInlet : FuelInletPort;
        port engineFuelReturn : FuelOutletPort;
    }

    part def FuelSystemTop {
        attribute systemName : String;
        attribute systemVersion : String;

        part fuelTankAssembly : FuelTankAssembly;
        part engine : Engine;

        connect fuelTankAssembly.assemblyOutlet to engine.engineFuelInlet;
        connect engine.engineFuelReturn to fuelTankAssembly.assemblyReturn;
    }
}