package AutomotiveFuelSystem {
    private import ScalarValues::*;

    enum def FuelFlowDirection {
        enum supply;
        enum return;
    }

    part def FuelTankAssembly {
        attribute capacity : Real;
        attribute currentFuelLevel : Real;
        attribute fuelTemperature : Real;

        port fuelOutlet : FuelPort;
        port fuelReturn : FuelPort;
    }

    part def FuelPump {
        attribute isActive : Boolean;
        attribute flowRate : Real;

        port fuelInlet : FuelPort;
        port fuelOutlet : FuelPort;
    }

    part def FuelPort {
        attribute direction : FuelFlowDirection;
    }

    item def Fuel {
        attribute temperature : Real;
    }

    part fuelSystem :> FuelTankAssembly {
        part pump : FuelPump;

        connect fuelOutlet to pump.fuelInlet;
        connect pump.fuelOutlet to fuelOutlet;
        connect fuelReturn to pump.fuelInlet;
    }
}