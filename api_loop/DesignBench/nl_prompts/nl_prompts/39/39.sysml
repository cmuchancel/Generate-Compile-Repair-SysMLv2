package 'VehiclePowerManagement39' {
	private import ScalarValues::*;

	part def VehiclePowerManagementSystem {
		part powerStateController : PowerStateController;
		part healthStateManager : HealthStateManager;
		attribute accurateSwitchingEnsured : Boolean;
		attribute promptSwitchingEnsured : Boolean;
		attribute dailyUseNeedsSupported : Boolean;
	}

	part def PowerStateController {
		attribute offState : Boolean;
		attribute startState : Boolean;
		attribute onState : Boolean;
		attribute startSignal : ControlSignal;
		attribute onSignal : ControlSignal;
		attribute offSignal : ControlSignal;
		attribute initialOffMode : Boolean;
	}

	part def HealthStateManager {
		attribute healthStateManagementReserved : Boolean;
	}

	attribute def ControlSignal {
		attribute signalActive : Boolean;
	}
}
