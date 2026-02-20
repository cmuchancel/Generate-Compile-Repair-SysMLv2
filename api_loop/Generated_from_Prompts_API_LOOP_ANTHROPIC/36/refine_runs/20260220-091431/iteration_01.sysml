package VehicleStateManagement {
    private import ScalarValues::*;

    enum def VehicleState {
        enum off;
        enum starting;
        enum on;
    }

    enum def ControlSignal {
        enum VehicleStartSignal;
        enum VehicleOnSignal;
        enum VehicleOffSignal;
    }

    part def VehicleController {
        attribute currentState : VehicleState = VehicleState::off;
        attribute lastSignal : ControlSignal;
    }

    part def VehicleSystem {
        attribute vehicleId : String;
        attribute initialState : VehicleState = VehicleState::off;
        attribute activeState : VehicleState = VehicleState::off;
        attribute currentSignal : ControlSignal;

        part controller : VehicleController;
    }

    part def StateTransitionRecord {
        attribute fromState : VehicleState;
        attribute toState : VehicleState;
        attribute triggerSignal : ControlSignal;
    }

    part def OffState {
        attribute stateId : VehicleState = VehicleState::off;
        attribute description : String = "Vehicle is powered off";
        attribute transitionSignal : ControlSignal = ControlSignal::VehicleStartSignal;
    }

    part def StartingState {
        attribute stateId : VehicleState = VehicleState::starting;
        attribute description : String = "Vehicle is in starting process";
        attribute transitionSignal : ControlSignal = ControlSignal::VehicleOnSignal;
    }

    part def OnState {
        attribute stateId : VehicleState = VehicleState::on;
        attribute description : String = "Vehicle is running";
        attribute transitionSignal : ControlSignal = ControlSignal::VehicleOffSignal;
    }

    port def SignalInputPort {
        attribute signalType : ControlSignal;
    }

    port def StateOutputPort {
        attribute stateValue : VehicleState;
    }

    part def SignalProcessor {
        attribute processorId : String;
        attribute incomingSignal : ControlSignal;
        attribute outgoingState : VehicleState;

        port signalIn : SignalInputPort;
        port stateOut : StateOutputPort;
    }

    part vehicleSystem : VehicleSystem {
        part offStateHandler : OffState;
        part startingStateHandler : StartingState;
        part onStateHandler : OnState;
        part transitionRecord : StateTransitionRecord;
        part signalProcessor : SignalProcessor;
    }

    part def VehicleMonitor {
        attribute monitorId : String;
        attribute observedState : VehicleState;
        attribute observedSignal : ControlSignal;

        port monitorStatePort : StateOutputPort;
    }

    part vehicleMonitor : VehicleMonitor;

    connect vehicleSystem.signalProcessor.stateOut to vehicleMonitor.monitorStatePort;
}