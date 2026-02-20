package FuelSystemManagement {
    private import ScalarValues::*;

    enum def FlowDirection {
        enum SUPPLY;
        enum RETURN;
    }

    item def FuelItem {
        attribute temperature : Real;
        attribute flowDirection : FlowDirection;
    }

    port def FuelPort {
        attribute temperature : Real;
        out item fuelOut : FuelItem;
        in item fuelIn : FuelItem;
    }

    part def FuelTankAssembly {
        attribute tankTemperature : Real;
        attribute fuelLevel : Real;
        port supplyPort : FuelPort;
        port returnPort : FuelPort;
    }

    part def Engine {
        attribute engineTemperature : Real;
        attribute fuelConsumptionRate : Real;
        port fuelSupplyPort : FuelPort;
        port fuelReturnPort : FuelPort;
    }

    part def VehicleFuelSystem {
        attribute systemStatus : String;

        part fuelTank : FuelTankAssembly {
            attribute tankTemperature : Real = 25.0;
            attribute fuelLevel : Real = 100.0;
            port supplyPort : FuelPort {
                attribute temperature : Real = 25.0;
            }
            port returnPort : FuelPort {
                attribute temperature : Real = 25.0;
            }
        }

        part engine : Engine {
            attribute engineTemperature : Real = 90.0;
            attribute fuelConsumptionRate : Real = 10.0;
            port fuelSupplyPort : FuelPort {
                attribute temperature : Real = 25.0;
            }
            port fuelReturnPort : FuelPort {
                attribute temperature : Real = 60.0;
            }
        }

        connect fuelTank.supplyPort to engine.fuelSupplyPort;
        connect engine.fuelReturnPort to fuelTank.returnPort;
    }
}