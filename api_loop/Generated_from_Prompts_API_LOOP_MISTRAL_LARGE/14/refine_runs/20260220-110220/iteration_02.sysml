package AutomotiveFuelSystem {
    private import ScalarValues::*;

    enum def FuelFlowDirection {
        enum supplyFlow;
        enum returnFlow;
    }

    port def FuelPort {
        attribute direction : FuelFlowDirection;
    }

    item def Fuel {
        attribute temperature : Real;
    }

    part def FuelTankAssembly {
        attribute capacity : Real;
        attribute currentFuelLevel : Real;
        attribute fuelTemperature : Real;

        port fuelSupplyOutlet : FuelPort;
        port fuelReturnInlet : FuelPort;
    }

    part def FuelPump {
        attribute isActive : Boolean;
        attribute flowRate : Real;

        port fuelInlet : FuelPort;
        port fuelOutlet : FuelPort;
    }

    part fuelSystem : FuelTankAssembly {
        part pump : FuelPump;

        connect fuelSupplyOutlet to pump.fuelInlet;
        connect pump.fuelOutlet to fuelSupplyOutlet;
        connect fuelReturnInlet to pump.fuelInlet;
    }
}