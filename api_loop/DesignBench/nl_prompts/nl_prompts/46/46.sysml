package 'CruiseControlSystem46' {
	private import ScalarValues::*;

	part def CruiseControlSystem {
		part driver : Driver;
		part cruiseController : CruiseController;
		part speedometer : Speedometer;
		part engine : Engine;
		attribute messageTransmissionUsed : Boolean;
		attribute acceleratorOperationReduced : Boolean;
	}

	part def Driver {
		port setSpeedPort : SetSpeedPort;
		attribute targetSpeedSelected : Boolean;
	}

	part def CruiseController {
		port setSpeedInput : SetSpeedPort;
		port speedInput : SpeedFeedbackPort;
		port throttleCommandOutput : ThrottleControlPort;
		attribute speedDifferenceComputed : Boolean;
	}

	part def Speedometer {
		port speedOutput : SpeedFeedbackPort;
		attribute vehicleSpeedProvided : Boolean;
	}

	part def Engine {
		port throttleInput : ThrottleControlPort;
		attribute engineOutputAdjusted : Boolean;
	}

	port def SetSpeedPort {
		attribute setSpeed : SpeedValue;
	}

	port def SpeedFeedbackPort {
		attribute actualSpeed : SpeedValue;
	}

	port def ThrottleControlPort {
		attribute throttleControlCommand : ControlValue;
	}

	attribute def SpeedValue {
		attribute magnitude : Real;
	}

	attribute def ControlValue {
		attribute magnitude : Real;
	}

	connection def DriverToController {
		end fromDriver : Driver::setSpeedPort;
		end toController : CruiseController::setSpeedInput;
	}

	connection def SpeedometerToController {
		end fromSpeedometer : Speedometer::speedOutput;
		end toController : CruiseController::speedInput;
	}

	connection def ControllerToEngine {
		end fromController : CruiseController::throttleCommandOutput;
		end toEngine : Engine::throttleInput;
	}
}
