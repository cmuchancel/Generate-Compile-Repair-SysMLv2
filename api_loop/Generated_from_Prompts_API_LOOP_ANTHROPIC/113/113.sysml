package VehicleSystem {
    private import ScalarValues::*;

    enum def VehicleStatus {
        enum SHUTDOWN;
        enum STARTUP;
        enum NORMAL;
        enum DEGRADED;
    }

    enum def FuelType {
        enum GASOLINE;
        enum DIESEL;
        enum ELECTRIC;
        enum HYBRID;
    }

    item def StartSignal;
    item def BrakePedalSignal;
    item def OverheatWarning;
    item def ShutdownSignal;
    item def HealthStatusSignal;
    item def MaintenanceSignal;
    item def FaultRecoverySignal;
    item def ControlSignal;

    port def VehicleOutputPort {
        out attribute statusOut : VehicleStatus;
        out attribute overheatWarningOut : Boolean;
        out attribute healthStatusOut : Boolean;
        out attribute maintenanceRequestOut : Boolean;
        out attribute faultRecoveryOut : Boolean;
        out attribute currentTemperatureOut : Real;
    }

    port def VehicleInputPort {
        in attribute startSignalIn : Boolean;
        in attribute brakePedalIn : Boolean;
        in attribute shutdownSignalIn : Boolean;
        in attribute controlSignalIn : Boolean;
    }

    port def ControllerOutputPort {
        out attribute startSignalOut : Boolean;
        out attribute shutdownSignalOut : Boolean;
        out attribute controlSignalOut : Boolean;
    }

    port def ControllerInputPort {
        in attribute statusIn : VehicleStatus;
        in attribute overheatWarningIn : Boolean;
        in attribute healthStatusIn : Boolean;
        in attribute maintenanceRequestIn : Boolean;
        in attribute faultRecoveryIn : Boolean;
        in attribute currentTemperatureIn : Real;
    }

    part def Vehicle {
        attribute vehicleId : String;
        attribute fuelType : FuelType;
        attribute maintenanceIntervalKm : Real;
        attribute temperatureThreshold : Real;
        attribute currentTemperature : Real;
        attribute status : VehicleStatus;
        attribute brakePedalPressed : Boolean;
        attribute startSignalReceived : Boolean;
        attribute shutdownSignalReceived : Boolean;
        attribute overheatDetected : Boolean;
        attribute selfInspectionPassed : Boolean;
        attribute powerSupplyReady : Boolean;
        attribute maintenanceDue : Boolean;
        attribute faultDetected : Boolean;
        attribute faultRecovered : Boolean;
        attribute kmSinceLastMaintenance : Real;
        attribute healthStatus : Boolean;

        port vehicleOut : VehicleOutputPort;
        port vehicleIn : VehicleInputPort;
    }

    part def VehicleController {
        attribute controllerId : String;
        attribute autoStartEnabled : Boolean;
        attribute autoShutdownEnabled : Boolean;
        attribute receivedStatus : VehicleStatus;
        attribute receivedOverheatWarning : Boolean;
        attribute receivedHealthStatus : Boolean;
        attribute receivedMaintenanceRequest : Boolean;
        attribute receivedFaultRecovery : Boolean;
        attribute receivedTemperature : Real;
        attribute startCommandIssued : Boolean;
        attribute shutdownCommandIssued : Boolean;

        port controllerOut : ControllerOutputPort;
        port controllerIn : ControllerInputPort;
    }

    part def VehicleSystem {
        attribute systemName : String;
        attribute systemVersion : String;
        attribute fuelConfiguration : FuelType;
        attribute globalMaintenanceInterval : Real;
        attribute globalTemperatureThreshold : Real;

        part vehicle : Vehicle {
            attribute vehicleId = "V-001";
            attribute fuelType = FuelType::GASOLINE;
            attribute maintenanceIntervalKm = 10000.0;
            attribute temperatureThreshold = 95.0;
            attribute currentTemperature = 20.0;
            attribute status = VehicleStatus::SHUTDOWN;
            attribute brakePedalPressed = false;
            attribute startSignalReceived = false;
            attribute shutdownSignalReceived = false;
            attribute overheatDetected = false;
            attribute selfInspectionPassed = false;
            attribute powerSupplyReady = false;
            attribute maintenanceDue = false;
            attribute faultDetected = false;
            attribute faultRecovered = false;
            attribute kmSinceLastMaintenance = 0.0;
            attribute healthStatus = true;
        }

        part controller : VehicleController {
            attribute controllerId = "VC-001";
            attribute autoStartEnabled = true;
            attribute autoShutdownEnabled = true;
            attribute receivedOverheatWarning = false;
            attribute receivedHealthStatus = true;
            attribute receivedMaintenanceRequest = false;
            attribute receivedFaultRecovery = false;
            attribute receivedTemperature = 20.0;
            attribute startCommandIssued = false;
            attribute shutdownCommandIssued = false;
        }

        connect vehicle.vehicleOut to controller.controllerIn;
        connect controller.controllerOut to vehicle.vehicleIn;
    }
}