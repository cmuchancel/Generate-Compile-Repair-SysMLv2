package 'CruiseControlSystem47' {
	private import ScalarValues::*;

	part def CruiseControlSystem {
		part driver : Driver;
		part cruiseController : CruiseController;
		part speedometer : Speedometer;
		part engine : Engine;
		attribute convenienceEnsured : Boolean;
		attribute comfortEnsured : Boolean;
		attribute accuracyEnsured : Boolean;
		attribute smoothnessEnsured : Boolean;
	}

	part def Driver {
		port setSpeedPort : SetSpeedPort;
		attribute speedSettingRequested : Boolean;
	}

	part def CruiseController {
		port setSpeedInput : SetSpeedPort;
		port speedFeedback : SpeedFeedbackPort;
		port throttleCommand : ThrottleControlPort;
		attribute throttleAdjustmentIntelligent : Boolean;
	}

	part def Speedometer {
		port speedOutput : SpeedFeedbackPort;
		attribute realTimeSpeedFeedbackProvided : Boolean;
	}

	part def Engine {
		port throttleInput : ThrottleControlPort;
		attribute automaticCruiseMaintained : Boolean;
	}

	port def SetSpeedPort {
		attribute desiredSpeed : SpeedValue;
	}

	port def SpeedFeedbackPort {
		attribute sensedSpeed : SpeedValue;
	}

	port def ThrottleControlPort {
		attribute throttleInstruction : ControlValue;
	}

	attribute def SpeedValue {
		attribute magnitude : Real;
	}

	attribute def ControlValue {
		attribute value : Real;
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
		end toEngine : Engine::throttleInput;
	}
}
