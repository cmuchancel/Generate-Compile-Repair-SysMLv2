package 'VehicleStateAutomation36' {
	private import ScalarValues::*;

	part def VehicleStateAutomationSystem {
		part vehicleLifecycleController : VehicleLifecycleController;
		attribute automaticStateManagement : Boolean;
		attribute comprehensiveControlProvided : Boolean;
	}

	part def VehicleLifecycleController {
		attribute initialOffState : Boolean;
		attribute vehicleStartSignal : ControlSignal;
		attribute vehicleOnSignal : ControlSignal;
		attribute vehicleOffSignal : ControlSignal;
		attribute currentState : VehicleLifecycleState;
		attribute shutdownToStartManaged : Boolean;
		attribute startToRunManaged : Boolean;
		attribute runToShutdownManaged : Boolean;
	}

	attribute def VehicleLifecycleState {
		attribute offState : Boolean;
		attribute startingState : Boolean;
		attribute onState : Boolean;
	}

	attribute def ControlSignal {
		attribute signalActive : Boolean;
	}
}
