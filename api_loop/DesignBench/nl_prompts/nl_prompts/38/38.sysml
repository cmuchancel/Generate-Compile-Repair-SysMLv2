package 'VehicleStartStopControl38' {
	private import ScalarValues::*;

	part def VehicleStartStopControlSystem {
		part startStopController : StartStopController;
		attribute automaticStateSwitching : Boolean;
		attribute convenientControlProvided : Boolean;
		attribute safeControlProvided : Boolean;
	}

	part def StartStopController {
		attribute shutdownState : Boolean;
		attribute startupState : Boolean;
		attribute runningState : Boolean;
		attribute vehicleStartSignal : ControlSignal;
		attribute vehicleStartedSignal : ControlSignal;
		attribute vehicleShutdownSignal : ControlSignal;
	}

	attribute def ControlSignal {
		attribute signalActive : Boolean;
	}
}
