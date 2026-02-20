package 'CruiseControlSystem48' {
	private import ScalarValues::*;

	part def CruiseControlSystem {
		part cruiseInterface : CruiseInterface;
		part cruiseController : CruiseController;
		part speedSensor : SpeedSensor;
		part engine : Engine;
		part vehicle : Vehicle;
		attribute simpleDriverOperation : Boolean;
		attribute reliableSystemOperation : Boolean;
		attribute automaticCruiseRequirementMet : Boolean;
	}

	part def CruiseInterface {
		port setSpeedPort : DataPort;
		attribute driverTargetSpeedIssued : Boolean;
	}

	part def CruiseController {
		port setSpeedInput : DataPort;
		port speedFeedback : DataPort;
		port fuelControlOutput : DataPort;
		attribute speedComparisonPerformed : Boolean;
	}

	part def SpeedSensor {
		port speedOutput : DataPort;
		attribute realTimeSpeedDetected : Boolean;
	}

	part def Engine {
		port fuelControlInput : DataPort;
		attribute engineOutputAdjusted : Boolean;
	}

	part def Vehicle {
		attribute automaticSpeedMaintenance : Boolean;
	}

	port def DataPort {
		attribute payload : DataPayload;
	}

	attribute def DataPayload {
		attribute value : Real;
	}

	connection def InterfaceToController {
		end fromInterface : CruiseInterface::setSpeedPort;
		end toController : CruiseController::setSpeedInput;
	}

	connection def SensorToController {
		end fromSensor : SpeedSensor::speedOutput;
		end toController : CruiseController::speedFeedback;
	}

	connection def ControllerToEngine {
		end fromController : CruiseController::fuelControlOutput;
		end toEngine : Engine::fuelControlInput;
	}
}
