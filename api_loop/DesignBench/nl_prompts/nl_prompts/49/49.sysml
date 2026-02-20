package 'CruiseControlSystem49' {
	private import ScalarValues::*;

	part def CruiseControlSystem {
		part driver : Driver;
		part cruiseController : CruiseController;
		part speedSensor : SpeedSensor;
		part engineController : EngineController;
		part engine : Engine;
		attribute standardizedMessageInterfaces : Boolean;
		attribute reliableOperationEnsured : Boolean;
		attribute efficientOperationEnsured : Boolean;
		attribute safeOperationEnsured : Boolean;
	}

	part def Driver {
		port setSpeedPort : MessagePort;
		attribute setSpeedCommandSent : Boolean;
	}

	part def CruiseController {
		port setSpeedInput : MessagePort;
		port speedInput : MessagePort;
		port fuelControlOutput : MessagePort;
		attribute differenceCalculated : Boolean;
	}

	part def SpeedSensor {
		port speedOutput : MessagePort;
		attribute realTimeSpeedReported : Boolean;
	}

	part def EngineController {
		port fuelControlInput : MessagePort;
		port forwardedFuelCommand : MessagePort;
		attribute processingPerformed : Boolean;
	}

	part def Engine {
		port fuelControlPort : MessagePort;
		attribute fuelFlowControlled : Boolean;
	}

	port def MessagePort {
		attribute messagePayload : MessagePayload;
	}

	attribute def MessagePayload {
		attribute value : Real;
	}

	connection def DriverToCruiseController {
		end fromDriver : Driver::setSpeedPort;
		end toCruiseController : CruiseController::setSpeedInput;
	}

	connection def SpeedSensorToCruiseController {
		end fromSensor : SpeedSensor::speedOutput;
		end toCruiseController : CruiseController::speedInput;
	}

	connection def CruiseControllerToEngineController {
		end fromCruiseController : CruiseController::fuelControlOutput;
		end toEngineController : EngineController::fuelControlInput;
	}

	connection def EngineControllerToEngine {
		end fromEngineController : EngineController::forwardedFuelCommand;
		end toEngine : Engine::fuelControlPort;
	}
}
