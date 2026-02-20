package 'VehicleStartStopControl42' {
	private import ScalarValues::*;

	part def VehicleStartStopSystem {
		part vehicleStateController : VehicleStateController;
		part brakeMonitor : BrakeMonitor;
		part controllerInterface : ControllerInterface;
		part powerSupplyModule : PowerSupplyModule;
		part selfCheckModule : SelfCheckModule;
		part parkingBrakeModule : ParkingBrakeModule;
		attribute safetyEnsured : Boolean;
		attribute reliabilityEnsured : Boolean;
		attribute coordinatedControlProvided : Boolean;
	}

	part def VehicleStateController {
		attribute offState : Boolean;
		attribute startProcessState : Boolean;
		attribute operatingState : Boolean;
		attribute startSignalReceived : Boolean;
		attribute powerOnSignalReceived : Boolean;
		attribute stopSignalReceived : Boolean;
		attribute startSignalSentToController : Boolean;
	}

	part def BrakeMonitor {
		attribute brakePedalDepressed : Boolean;
		attribute powerOnPermissionGranted : Boolean;
	}

	part def ControllerInterface {
		attribute startCommandIssued : Boolean;
	}

	part def PowerSupplyModule {
		attribute continuousPowerProvided : Boolean;
	}

	part def SelfCheckModule {
		attribute selfCheckDuringOperation : Boolean;
	}

	part def ParkingBrakeModule {
		attribute automaticApplication : Boolean;
		attribute parkingSafetyEnsured : Boolean;
	}
}
