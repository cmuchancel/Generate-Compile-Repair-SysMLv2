package FuelDeliverySystem {
    private import ScalarValues::*;

    item def Fuel;

    port def FuelOutPort {
        out item fuelOut : Fuel;
    }

    port def FuelInPort {
        in item fuelIn : Fuel;
    }

    part def StorageTank {
        attribute capacity : Real;
        attribute currentLevel : Real;
        port supplyPort : FuelOutPort;
    }

    part def FuelPump {
        port pumpInPort : FuelInPort;
        port pumpOutPort : FuelOutPort;
    }

    part def VehicleFuelTank {
        attribute maxCapacity : Real;
        attribute currentFuelLevel : Real;
        port receivePort : FuelInPort;
    }

    part def Vehicle {
        part vehicleFuelTank : VehicleFuelTank;
    }

    part def FuelDeliveryServiceSystem {
        part storageTank : StorageTank {
            attribute capacity = 10000.0;
            attribute currentLevel = 8000.0;
        }
        part fuelPump : FuelPump;
        part vehicle : Vehicle {
            part vehicleFuelTank : VehicleFuelTank {
                attribute maxCapacity = 60.0;
                attribute currentFuelLevel = 10.0;
            }
        }

        connect storageTank.supplyPort to fuelPump.pumpInPort;
        connect fuelPump.pumpOutPort to vehicle.vehicleFuelTank.receivePort;
    }
}