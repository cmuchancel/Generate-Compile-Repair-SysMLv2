package 'AutomaticBrakingSystem26' {
	private import ScalarValues::*;

	part def AutomaticBrakingSystem {
		part keyInterface : KeyStartInterface;
		part brakePedalMonitor : BrakePedalMonitor;
		part tractionMonitor : TractionMonitor;
		part brakeController : BrakeController;
		attribute automaticActivation : Boolean;
		attribute continuousMonitoring : Boolean;
		attribute safeBrakingEnsured : Boolean;
		attribute reliableBrakingEnsured : Boolean;
		attribute drivingSafetyEnhanced : Boolean;
		attribute drivingComfortEnhanced : Boolean;
		attribute optimalStateMaintained : Boolean;
	}

	part def KeyStartInterface {
		port keyPositionPort : KeyPositionPort;
		attribute startPositionDetected : Boolean;
	}

	part def BrakePedalMonitor {
		port brakePressurePort : BrakePressurePort;
		attribute pedalStatusMonitored : Boolean;
	}

	part def TractionMonitor {
		port tractionPort : TractionPort;
		attribute roadAdhesionEvaluated : Boolean;
	}

	part def BrakeController {
		port keyEnablePort : KeyPositionPort;
		port pedalPressureInput : BrakePressurePort;
		port tractionInput : TractionPort;
		port pressureAdjustment : BrakePressurePort;
		port modulationControlPort : ModulationPort;
		attribute brakePressureAdjusted : Boolean;
		attribute modulationFrequencyAdjusted : Boolean;
	}

	port def KeyPositionPort {
		attribute keyPosition : KeyPosition;
	}

	port def BrakePressurePort {
		attribute brakePressure : PressureMeasurement;
	}

	port def TractionPort {
		attribute tractionLevel : TractionMeasurement;
	}

	port def ModulationPort {
		attribute modulationFrequency : FrequencyMeasurement;
	}

	attribute def KeyPosition;
	attribute def PressureMeasurement {
		attribute pressureValue : Real;
	}
	attribute def TractionMeasurement {
		attribute adhesionValue : Real;
	}
	attribute def FrequencyMeasurement {
		attribute frequencyValue : Real;
	}

	connection def KeyActivation {
		end fromKey : KeyStartInterface::keyPositionPort;
		end toController : BrakeController::keyEnablePort;
	}

	connection def BrakePressureFeedback {
		end fromMonitor : BrakePedalMonitor::brakePressurePort;
		end toController : BrakeController::pedalPressureInput;
	}

	connection def TractionFeedback {
		end fromTraction : TractionMonitor::tractionPort;
		end toController : BrakeController::tractionInput;
	}
}
