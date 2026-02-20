package 'CruiseControlSystem44' {
	private import ScalarValues::*;

	part def CruiseControlSystem {
		part driver : Driver;
		part cruiseController : CruiseController;
		part speedometer : Speedometer;
		part engine : Engine;
		part vehicle : Vehicle;
		attribute eventBasedExchangeSupported : Boolean;
	}

	part def Driver {
		port setSpeedPort : SetSpeedPort;
		attribute desiredSpeedIssued : Boolean;
	}

	part def CruiseController {
		port setSpeedInput : SetSpeedPort;
		port speedInput : SpeedFeedbackPort;
		port fuelCommandOutput : FuelControlPort;
		attribute setSpeedCommandReceived : Boolean;
		attribute actualSpeedProcessed : Boolean;
	}

	part def Speedometer {
		port speedOutput : SpeedFeedbackPort;
		attribute realTimeSpeedCollected : Boolean;
	}

	part def Engine {
		port fuelControlInput : FuelControlPort;
		attribute outputAdjusted : Boolean;
	}

	part def Vehicle;

	port def SetSpeedPort {
		attribute setSpeedValue : SpeedValue;
	}

	port def SpeedFeedbackPort {
		attribute actualSpeedValue : SpeedValue;
	}

	port def FuelControlPort {
		attribute fuelControlCommand : FuelControlValue;
	}

	attribute def SpeedValue {
		attribute magnitude : Real;
	}

	attribute def FuelControlValue {
		attribute controlMagnitude : Real;
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
		end fromController : CruiseController::fuelCommandOutput;
		end toEngine : Engine::fuelControlInput;
	}
}
