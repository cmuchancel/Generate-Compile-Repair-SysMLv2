package FuelFlowManagement {
    private import ScalarValues::*;

    enum def FlowDirection {
        enum supply;
        enum return;
    }

    part def FuelTankAssembly {
        attribute capacity : Real;
        attribute currentLevel : Real;
        port supplyPort : FuelPort;
        port returnPort : FuelPort;
        attribute temperatureSensor : TemperatureSensor;
    }

    part def Engine {
        port fuelInlet : FuelPort;
        port fuelReturn : FuelPort;
        attribute temperatureSensor : TemperatureSensor;
    }

    part def TemperatureSensor {
        attribute currentTemp : Real;
    }

    port def FuelPort {
        attribute direction : FlowDirection;
        attribute flowRate : Real;
    }

    item def Fuel {
        attribute type : String;
    }

    part fuelTank : FuelTankAssembly;
    part engine : Engine;

    connect fuelTank.supplyPort to engine.fuelInlet;
    connect engine.fuelReturn to fuelTank.returnPort;
}