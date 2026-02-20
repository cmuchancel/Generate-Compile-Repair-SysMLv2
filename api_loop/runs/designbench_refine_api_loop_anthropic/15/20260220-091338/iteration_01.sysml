package FuelSystem {
    private import ScalarValues::*;

    enum def FuelFlowDirection {
        enum SUPPLY;
        enum RETURN;
    }

    item def Fuel;
    item def FuelTemperatureSignal;

    port def FuelSupplyPort {
        attribute flowDirection : FuelFlowDirection;
    }

    port def FuelReturnPort {
        attribute flowDirection : FuelFlowDirection;
    }

    port def TemperatureMonitorPort {
        attribute temperature : Real;
    }

    part def FuelTankAssembly {
        attribute fuelLevel : Real;
        attribute tankCapacity : Real;

        port supplyOut : FuelSupplyPort;
        port returnIn : FuelReturnPort;
        port tempMonitor : TemperatureMonitorPort;
    }

    part def Engine {
        attribute fuelConsumptionRate : Real;
        attribute operatingTemperature : Real;

        port supplyIn : FuelSupplyPort;
        port returnOut : FuelReturnPort;
        port tempSensor : TemperatureMonitorPort;
    }

    part def FuelLine {
        attribute lineLength : Real;
        attribute lineDiameter : Real;

        port lineInlet : FuelSupplyPort;
        port lineOutlet : FuelSupplyPort;
    }

    part def ReturnFuelLine {
        attribute lineLength : Real;
        attribute lineDiameter : Real;

        port returnLineInlet : FuelReturnPort;
        port returnLineOutlet : FuelReturnPort;
    }

    part def VehicleFuelSystem {
        attribute vehicleId : String;
        attribute systemOperational : Boolean;

        part tankAssembly : FuelTankAssembly;
        part engine : Engine;
        part supplyLine : FuelLine;
        part returnLine : ReturnFuelLine;

        connect tankAssembly.supplyOut to supplyLine.lineInlet;
        connect supplyLine.lineOutlet to engine.supplyIn;
        connect engine.returnOut to returnLine.returnLineInlet;
        connect returnLine.returnLineOutlet to tankAssembly.returnIn;
        connect tankAssembly.tempMonitor to engine.tempSensor;
    }
}