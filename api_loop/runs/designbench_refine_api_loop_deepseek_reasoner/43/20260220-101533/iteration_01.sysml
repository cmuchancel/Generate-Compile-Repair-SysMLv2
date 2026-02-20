package VehicleStartStopSystem {
    private import ScalarValues::*;
    
    enum def VehicleState {
        Off,
        Starting,
        Started,
        ShuttingDown
    }
    
    part def VehicleStartStopController {
        port systemInput : SystemInputPort;
        port systemOutput : SystemOutputPort;
        attribute currentState : VehicleState = VehicleState::Off;
        attribute brakePedalDepressed : Boolean = false;
        attribute parkingBrakeApplied : Boolean = false;
        attribute vehicleOnSignal : Boolean = false;
        attribute startControlSignal : Boolean = false;
        attribute shutdownSignal : Boolean = false;
        attribute systemActive : Boolean = false;
    }
    
    part def VehicleController {
        port controlInput : ControlInputPort;
        port statusOutput : StatusOutputPort;
        attribute vehicleStarted : Boolean = false;
        attribute powerSupplied : Boolean = false;
        attribute selfCheckComplete : Boolean = false;
    }
    
    part def BrakePedal {
        port brakeOutput : BrakeOutputPort;
        attribute pedalPosition : Real;
    }
    
    part def ParkingBrake {
        port brakeControl : BrakeControlPort;
        attribute brakeApplied : Boolean = false;
    }
    
    port def SystemInputPort {
        attribute startSignal : Boolean;
        attribute vehicleOn : Boolean;
        attribute shutdownRequest : Boolean;
    }
    
    port def SystemOutputPort {
        attribute startCommand : Boolean;
        attribute shutdownCommand : Boolean;
        attribute applyParkingBrake : Boolean;
    }
    
    port def ControlInputPort {
        attribute startVehicle : Boolean;
        attribute stopVehicle : Boolean;
    }
    
    port def StatusOutputPort {
        attribute vehicleStatus : Boolean;
        attribute powerStatus : Boolean;
        attribute selfCheckStatus : Boolean;
    }
    
    port def BrakeOutputPort {
        attribute pedalDepressed : Boolean;
    }
    
    port def BrakeControlPort {
        attribute applyBrake : Boolean;
        attribute releaseBrake : Boolean;
    }
    
    part mainController : VehicleStartStopController;
    part vehicleController : VehicleController;
    part brakePedal : BrakePedal;
    part parkingBrake : ParkingBrake;
    
    connect mainController.systemOutput to vehicleController.controlInput;
    connect mainController.systemOutput to parkingBrake.brakeControl;
    connect brakePedal.brakeOutput to mainController.systemInput;
    connect vehicleController.statusOutput to mainController.systemInput;
}