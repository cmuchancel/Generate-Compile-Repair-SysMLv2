package 'VehicleStartStopControl43' {
	private import ScalarValues::*;

	part def VehicleStartStopSystem {
		part vehicleStateController : VehicleStateController;
		part brakeValidation : BrakeValidation;
		part controllerInterface : ControllerInterface;
		part selfCheckModule : SelfCheckModule;
		part powerSupplyModule : PowerSupplyModule;
		part parkingBrakeModule : ParkingBrakeModule;
		attribute correctConditionEnforced : Boolean;
		attribute safetyImproved : Boolean;
		attribute convenienceImproved : Boolean;
	}

	part def VehicleStateController {
		attribute offState : Boolean;
		attribute startProcessState : Boolean;
		attribute startedState : Boolean;
		attribute startSignalReceived : Boolean;
		attribute vehicleOnSignalReceived : Boolean;
		attribute shutdownSignalReceived : Boolean;
		attribute startControlSignalIssued : Boolean;
	}

	part def BrakeValidation {
		attribute brakePedalDepressed : Boolean;
		attribute permissionToStart : Boolean;
	}

	part def ControllerInterface {
		attribute controllerStartCommanded : Boolean;
	}

	part def SelfCheckModule {
		attribute selfCheckActive : Boolean;
	}

	part def PowerSupplyModule {
		attribute continuousPowerSupplied : Boolean;
	}

	part def ParkingBrakeModule {
		attribute parkingBrakeAppliedOnShutdown : Boolean;
	}
}
