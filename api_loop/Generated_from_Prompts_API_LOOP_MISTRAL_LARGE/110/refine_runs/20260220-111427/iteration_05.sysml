package FuelDeliverySystem {
    private import ScalarValues::*;

    enum def FuelFlowState {
        enum STOPPED
        enum FLOWING
        enum BLOCKED
    }

    part def StorageTank {
        attribute maxCapacity : Real;
        attribute currentLevel : Real;
    }

    part def FuelPump {
        attribute flowRate : Real;
        attribute pumpState : FuelFlowState = FuelFlowState::STOPPED;
    }

    part def VehicleFuelTank {
        attribute maxCapacity : Real;
        attribute currentLevel : Real;
    }

    part def Vehicle {
        part fuelTank : VehicleFuelTank;
    }

    part def FuelDeliverySystem {
        part storageTank : StorageTank;
        part pump : FuelPump;
        part vehicle : Vehicle;

        connect storageTank to pump;
        connect pump to vehicle.fuelTank;
    }

    item def Fuel {
        attribute volume : Real;
    }
}