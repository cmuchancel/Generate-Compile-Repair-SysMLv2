package FuelDeliverySystem {
    private import ScalarValues::*;

    enum def FuelFlowState {
        enum IDLE;
        enum TRANSFERRING;
        enum COMPLETE;
    }

    part def StorageTank {
        attribute maxCapacity : Real;
        attribute currentLevel : Real;
        port fuelOutput : FuelPort;
    }

    part def FuelPump {
        port fuelInput : FuelPort;
        port fuelOutput : FuelPort;
        attribute isActive : Boolean;
    }

    part def VehicleFuelTank {
        attribute maxCapacity : Real;
        attribute currentLevel : Real;
        port fuelInput : FuelPort;
    }

    port def FuelPort;

    item def Fuel;

    part storageTank : StorageTank;
    part fuelPump : FuelPump;
    part vehicleFuelTank : VehicleFuelTank;

    connect storageTank.fuelOutput to fuelPump.fuelInput;
    connect fuelPump.fuelOutput to vehicleFuelTank.fuelInput;
}