package 'VehiclePowerManagement37' {
	private import ScalarValues::*;

	part def VehiclePowerManagementSystem {
		part powerStateController : PowerStateController;
		part selfDiagnosisModule : SelfDiagnosisModule;
		part powerDistributionUnit : PowerDistributionUnit;
		part safetyOperationModule : SafetyOperationModule;
		attribute intelligentControlProvided : Boolean;
		attribute stateManagementProvided : Boolean;
		attribute convenienceEnhanced : Boolean;
		attribute safetyEnhanced : Boolean;
	}

	part def PowerStateController {
		attribute offState : Boolean;
		attribute startPhaseState : Boolean;
		attribute poweredOnState : Boolean;
		attribute userStartCommand : ControlSignal;
		attribute vehicleStartedSignal : ControlSignal;
		attribute userShutdownCommand : ControlSignal;
	}

	part def SelfDiagnosisModule {
		attribute diagnosisDuringPoweredOn : Boolean;
	}

	part def PowerDistributionUnit {
		attribute continuousPowerSupply : Boolean;
		attribute functionalUnitsSupplied : Boolean;
	}

	part def SafetyOperationModule {
		attribute parkingBrakeEngagement : Boolean;
		attribute powerCutoffPerformed : Boolean;
	}

	attribute def ControlSignal {
		attribute signalActive : Boolean;
	}
}
