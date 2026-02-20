package FuelFlowManagement {
    private import ScalarValues::*;

    enum def FlowStatus {
        enum Active;
        enum Inactive;
    }

    item def Fuel;

    item def TemperatureData {
        attribute temperatureValue : Real;
    }

    item def FuelFlowData {
        attribute flowRate : Real;
        attribute status : FlowStatus;
    }

    port def FuelSupplyPort {
        attribute supplyFlowRate : Real;
        attribute supplyTemperature : Real;
    }

    port def FuelReturnPort {
        attribute returnFlowRate : Real;
        attribute returnTemperature : Real;
    }

    port def MonitoringPort {
        attribute supplyData : Real;
        attribute returnData : Real;
        attribute temperatureData : Real;
    }

    part def FuelTankAssembly {
        attribute tankCapacity : Real;
        attribute currentFuelLevel : Real;
        attribute fuelTemperature : Real;

        port fuelSupplyOut : FuelSupplyPort;
        port fuelReturnIn : FuelReturnPort;
        port monitorOut : MonitoringPort;
    }

    part def Engine {
        attribute fuelConsumptionRate : Real;
        attribute operatingTemperature : Real;

        port fuelSupplyIn : FuelSupplyPort;
        port fuelReturnOut : FuelReturnPort;
        port monitorOut : MonitoringPort;
    }

    part def FuelFlowManagementSystem {
        attribute systemStatus : FlowStatus;

        part fuelTank : FuelTankAssembly {
            attribute tankCapacity : Real = 500.0;
            attribute currentFuelLevel : Real = 400.0;
            attribute fuelTemperature : Real = 25.0;
        }

        part engine : Engine {
            attribute fuelConsumptionRate : Real = 10.0;
            attribute operatingTemperature : Real = 150.0;
        }

        connect fuelTank.fuelSupplyOut to engine.fuelSupplyIn;
        connect engine.fuelReturnOut to fuelTank.fuelReturnIn;
        connect fuelTank.monitorOut to engine.monitorOut;
    }
}