package FuelDeliverySystem {
    private import ScalarValues::*;

    part def StorageTank {
        attribute currentLevel : Real;
        port fuelOutlet : FuelPort;
    }

    part def FuelPump {
        port fuelInlet : FuelPort;
        port fuelOutlet : FuelPort;
    }

    part def VehicleFuelTank {
        attribute maxCapacity : Real;
        attribute currentLevel : Real;
        port fuelInlet : FuelPort;
    }

    port def FuelPort;

    part def FuelDeliverySystem {
        part storage : StorageTank;
        part pump : FuelPump;
        part vehicleTank : VehicleFuelTank;

        connect storage.fuelOutlet to pump.fuelInlet;
        connect pump.fuelOutlet to vehicleTank.fuelInlet;
    }
}