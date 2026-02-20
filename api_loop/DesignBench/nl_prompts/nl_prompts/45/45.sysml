package 'CruiseControlSystem45' {
	private import ScalarValues::*;

	part def CruiseControlSystem {
		part driver : Driver;
		part cruiseController : CruiseController;
		part speedometer : Speedometer;
		part engineSystem : EngineSystem;
		attribute automaticDriverResponse : Boolean;
		attribute comfortImproved : Boolean;
		attribute safetyImproved : Boolean;
	}

	part def Driver {
		port setSpeedPort : SetSpeedPort;
		attribute desiredSpeedCommanded : Boolean;
	}

	part def CruiseController {
		port setSpeedInput : SetSpeedPort;
		port speedFeedback : SpeedFeedbackPort;
		port throttleCommand : ThrottleControlPort;
		attribute setSpeedStored : Boolean;
		attribute speedDifferenceEvaluated : Boolean;
	}

	part def Speedometer {
		port speedOutput : SpeedFeedbackPort;
		attribute realTimeSpeedProvided : Boolean;
	}

	part def EngineSystem {
		port throttleInput : ThrottleControlPort;
		attribute powerOutputAdjusted : Boolean;
	}

	port def SetSpeedPort {
		attribute setSpeed : SpeedValue;
	}

	port def SpeedFeedbackPort {
		attribute actualSpeed : SpeedValue;
	}

	port def ThrottleControlPort {
		attribute throttleCommandValue : ControlValue;
	}

	attribute def SpeedValue {
		attribute magnitude : Real;
	}

	attribute def ControlValue {
		attribute amount : Real;
	}

	connection def DriverToController {
		end fromDriver : Driver::setSpeedPort;
		end toController : CruiseController::setSpeedInput;
	}

	connection def SpeedometerToController {
		end fromSpeedometer : Speedometer::speedOutput;
		end toController : CruiseController::speedFeedback;
	}

	connection def ControllerToEngine {
		end fromController : CruiseController::throttleCommand;
		end toEngine : EngineSystem::throttleInput;
	}
}
