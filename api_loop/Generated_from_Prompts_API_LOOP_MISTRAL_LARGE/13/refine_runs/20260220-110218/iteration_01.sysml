package FuelSystem {
    private import ScalarValues::*;

    enum def FuelFlowDirection {
        enum supply;
        enum return;
    }

    part def FuelTankAssembly {
        attribute capacity : Real;
        attribute currentFuelLevel : Real;
        part fuelPump : FuelPump;
        part fuelTank : FuelTank;
    }

    part def FuelPump {
        attribute isActive : Boolean;
        attribute pressure : Real;
        port fuelOut : FuelSupplyPort;
        port fuelReturn : FuelReturnPort;
    }

    part def FuelTank {
        attribute temperature : Real;
        port fuelIn : FuelReturnPort;
        port fuelOut : FuelSupplyPort;
    }

    part def FuelConsumingUnit {
        port fuelIn : FuelSupplyPort;
        port fuelReturn : FuelReturnPort;
    }

    port def FuelSupplyPort {
        attribute direction : FuelFlowDirection = FuelFlowDirection::supply;
    }

    port def FuelReturnPort {
        attribute direction : FuelFlowDirection = FuelFlowDirection::return;
    }

    item def Fuel {
        attribute temperature : Real;
    }

    part vehicleFuelSystem : FuelTankAssembly {
        part engine : FuelConsumingUnit;
        part auxiliaryUnit : FuelConsumingUnit;

        connect fuelTank.fuelOut to fuelPump.fuelOut;
        connect fuelPump.fuelOut to engine.fuelIn;
        connect engine.fuelReturn to fuelTank.fuelIn;
        connect fuelPump.fuelReturn to fuelTank.fuelIn;
        connect auxiliaryUnit.fuelReturn to fuelTank.fuelIn;
    }
}