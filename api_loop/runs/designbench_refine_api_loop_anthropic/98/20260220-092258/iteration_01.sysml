package FuelManagementSystem {
    private import ScalarValues::*;

    enum def FuelFlowState {
        enum Flowing;
        enum Stopped;
        enum Returning;
    }

    item def Fuel;

    port def FuelSupplyPort {
        attribute flowRate : Real;
        attribute temperature : Real;
    }

    port def FuelReturnPort {
        attribute flowRate : Real;
        attribute temperature : Real;
    }

    part def FuelTank {
        attribute capacity : Real;
        attribute currentLevel : Real;
        attribute fuelTemperature : Real;
        port supplyOut : FuelSupplyPort;
        port returnIn : FuelReturnPort;
    }

    part def Engine {
        attribute operatingTemperature : Real;
        attribute fuelConsumptionRate : Real;
        port supplyIn : FuelSupplyPort;
        port returnOut : FuelReturnPort;
    }

    part def FuelFlowController {
        attribute flowState : FuelFlowState;
        attribute monitoredTemperature : Real;
        attribute targetFlowRate : Real;
        port controlSupplyIn : FuelSupplyPort;
        port controlSupplyOut : FuelSupplyPort;
        port controlReturnIn : FuelReturnPort;
        port controlReturnOut : FuelReturnPort;
    }

    part def FuelManagementSystemAssembly {
        attribute systemName : String;
        attribute operationalStatus : Boolean;

        part tank : FuelTank {
            attribute capacity : Real = 500.0;
            attribute currentLevel : Real = 400.0;
            attribute fuelTemperature : Real = 25.0;
        }

        part engine : Engine {
            attribute operatingTemperature : Real = 90.0;
            attribute fuelConsumptionRate : Real = 10.0;
        }

        part controller : FuelFlowController {
            attribute flowState : FuelFlowState = FuelFlowState::Flowing;
            attribute monitoredTemperature : Real = 25.0;
            attribute targetFlowRate : Real = 10.0;
        }

        connect tank.supplyOut to controller.controlSupplyIn;
        connect controller.controlSupplyOut to engine.supplyIn;
        connect engine.returnOut to controller.controlReturnIn;
        connect controller.controlReturnOut to tank.returnIn;
    }
}