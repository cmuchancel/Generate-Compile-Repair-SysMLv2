package 'VehicleStateManagement35' {
	private import ScalarValues::*;

	part def VehicleStateManagementSystem {
		part vehicleStateController : VehicleStateController;
		attribute startupStateManaged : Boolean;
		attribute shutdownStateManaged : Boolean;
		attribute sequentialTransitionsEnsured : Boolean;
		attribute safeResponsesEnsured : Boolean;
	}

	part def VehicleStateController {
		attribute currentState : VehicleState;
		attribute vehicleStartSignal : ControlSignal;
		attribute vehiclePowerOnSignal : ControlSignal;
		attribute vehicleShutdownSignal : ControlSignal;
		attribute initialOffState : Boolean;
		attribute startTransitionSupported : Boolean;
		attribute powerOnTransitionSupported : Boolean;
		attribute shutdownTransitionSupported : Boolean;
	}

	attribute def VehicleState {
		attribute offState : Boolean;
		attribute startingState : Boolean;
		attribute onState : Boolean;
	}

	attribute def ControlSignal {
		attribute active : Boolean;
	}
}
