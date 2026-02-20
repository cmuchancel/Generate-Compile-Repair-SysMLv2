package FuelSupplySystem {
    private import ScalarValues::*;

    enum def FuelFlowState {
        enum Flowing;
        enum Idle;
        enum Returning;
    }

    item def FuelItem;
    item def FuelDataItem;

    port def FuelOutPort {
        out item fuel : FuelItem;
    }

    port def FuelInPort {
        in item fuel : FuelItem;
    }

    port def FuelReturnOutPort {
        out item returnFuel : FuelItem;
    }

    port def FuelReturnInPort {
        in item returnFuel : FuelItem;
    }

    port def TempDataOutPort {
        out item tempData : FuelDataItem;
    }

    port def TempDataInPort {
        in item tempData : FuelDataItem;
    }

    part def FuelTank {
        attribute capacity : Real;
        attribute currentFuelLevel : Real;
        attribute fuelTemperature : Real;

        port tankSupplyOut : FuelOutPort;
        port tankReturnIn : FuelReturnInPort;
        port tankTempOut : TempDataOutPort;
    }

    part def FuelPump {
        attribute pressure : Real;
        attribute flowRate : Real;
        attribute flowState : FuelFlowState;

        port pumpFuelIn : FuelInPort;
        port pumpFuelOut : FuelOutPort;
        port pumpReturnOut : FuelReturnOutPort;
        port pumpTempIn : TempDataInPort;
        port pumpTempOut : TempDataOutPort;
    }

    part def FuelTankAssembly {
        attribute assemblyId : String;

        part tank : FuelTank;
        part pump : FuelPump;

        port assemblySupplyOut : FuelOutPort;
        port assemblyReturnIn : FuelReturnInPort;
        port assemblyTempOut : TempDataOutPort;

        connect tank.tankSupplyOut to pump.pumpFuelIn;
        connect pump.pumpReturnOut to tank.tankReturnIn;
        connect tank.tankTempOut to pump.pumpTempIn;
        connect pump.pumpFuelOut to assemblySupplyOut;
        connect assemblyReturnIn to tank.tankReturnIn;
        connect pump.pumpTempOut to assemblyTempOut;
    }

    part def FuelConsumingUnit {
        attribute unitId : String;
        attribute fuelConsumptionRate : Real;

        port unitFuelIn : FuelInPort;
        port unitReturnOut : FuelReturnOutPort;
        port unitTempIn : TempDataInPort;
    }

    part def VehicleFuelSystem {
        attribute vehicleId : String;
        attribute systemStatus : FuelFlowState;

        part fuelAssembly : FuelTankAssembly;
        part engineUnit : FuelConsumingUnit;
        part auxiliaryUnit : FuelConsumingUnit;

        connect fuelAssembly.assemblySupplyOut to engineUnit.unitFuelIn;
        connect fuelAssembly.assemblySupplyOut to auxiliaryUnit.unitFuelIn;
        connect engineUnit.unitReturnOut to fuelAssembly.assemblyReturnIn;
        connect auxiliaryUnit.unitReturnOut to fuelAssembly.assemblyReturnIn;
        connect fuelAssembly.assemblyTempOut to engineUnit.unitTempIn;
        connect fuelAssembly.assemblyTempOut to auxiliaryUnit.unitTempIn;
    }
}