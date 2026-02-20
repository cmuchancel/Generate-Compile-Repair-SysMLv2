package VehicleFuelSystem {
    private import ScalarValues::*;

    enum def FuelFlowDirection {
        enum supply;
        enum return;
    }

    part def FuelTankAssembly {
        attribute capacity : Real;
        attribute currentLevel : Real;
        attribute fuelTemperature : Real;

        port fuelSupplyPort : FuelPort;
        port fuelReturnPort : FuelPort;
    }

    part def Engine {
        attribute fuelConsumptionRate : Real;
        attribute fuelTemperature : Real;

        port fuelSupplyPort : FuelPort;
        port fuelReturnPort : FuelPort;
    }

    part def FuelLine {
        attribute flowDirection : FuelFlowDirection;
        attribute fuelTemperature : Real;
    }

    port def FuelPort {
        attribute direction : FuelFlowDirection;
    }

    item def Fuel {
        attribute temperature : Real;
    }

    part vehicleFuelSystem {
        part fuelTank : FuelTankAssembly;
        part engine : Engine;
        part supplyLine : FuelLine;
        part returnLine : FuelLine;

        supplyLine.flowDirection = FuelFlowDirection::supply;
        returnLine.flowDirection = FuelFlowDirection::return;

        connect fuelTank.fuelSupplyPort to supplyLine;
        connect supplyLine to engine.fuelSupplyPort;
        connect engine.fuelReturnPort to returnLine;
        connect returnLine to fuelTank.fuelReturnPort;
    }
}