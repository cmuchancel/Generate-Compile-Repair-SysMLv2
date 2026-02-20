package FuelManagementSystem {
    private import ScalarValues::*;

    enum def FlowDirection {
        enum SUPPLY;
        enum RETURN;
    }

    item def FuelItem {
        attribute temperature : Real;
        attribute flowDirection : FlowDirection;
    }

    port def FuelOutletPort {
        attribute fuelTemperature : Real;
        out item fuelFlow : FuelItem;
    }

    port def FuelInletPort {
        attribute fuelTemperature : Real;
        in item fuelFlow : FuelItem;
    }

    port def FuelReturnOutPort {
        attribute fuelTemperature : Real;
        out item fuelFlow : FuelItem;
    }

    port def FuelReturnInPort {
        attribute fuelTemperature : Real;
        in item fuelFlow : FuelItem;
    }

    part def FuelTankAssembly {
        attribute tankCapacity : Real;
        attribute currentFuelLevel : Real;
        attribute fuelTemperature : Real;

        port fuelOutlet : FuelOutletPort;
        port fuelReturnIn : FuelReturnInPort;
    }

    part def EngineUnit {
        attribute operatingTemperature : Real;
        attribute fuelConsumptionRate : Real;

        port fuelInlet : FuelInletPort;
        port fuelReturnOut : FuelReturnOutPort;
    }

    part def VehicleFuelSystem {
        attribute systemStatus : String;

        part fuelTank : FuelTankAssembly {
            attribute tankCapacity : Real = 60.0;
            attribute currentFuelLevel : Real = 45.0;
            attribute fuelTemperature : Real = 25.0;
        }

        part engine : EngineUnit {
            attribute operatingTemperature : Real = 90.0;
            attribute fuelConsumptionRate : Real = 8.5;
        }

        connect fuelTank.fuelOutlet to engine.fuelInlet;
        connect engine.fuelReturnOut to fuelTank.fuelReturnIn;
    }
}