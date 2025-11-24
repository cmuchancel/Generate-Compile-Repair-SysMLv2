package 'VehicleCruiseControl' {
	private import ScalarValues::Real;
	
	item def SetSpeedCommand {
		attribute targetSpeed : Real;
	}
	
	item def SpeedSense {
		attribute actualSpeed : Real;
	}
	
	item def FuelControl {
		attribute throttleValue : Real;
	}
	
	port def SetSpeedPort {
		attribute command : SetSpeedCommand;
	}
	
	port def SpeedSensePort {
		attribute measurement : SpeedSense;
	}
	
	port def FuelControlPort {
		attribute control : FuelControl;
	}
	
	action def GenerateFuelControl {
		in setSpeed : SetSpeedCommand;
		in actualSpeed : SpeedSense;
		out control : FuelControl;
	}
	
	part driverInterface {
		port setSpeedPort : SetSpeedPort;
	}
	
	part vehicle {
		part cruiseController {
			port setSpeedPort : SetSpeedPort;
			port speedPort : SpeedSensePort;
			port fuelControlPort : FuelControlPort;
			perform action generateFuelControl : GenerateFuelControl {
				in setSpeed = setSpeedPort.command;
				in actualSpeed = speedPort.measurement;
				out control = fuelControlPort.control;
			}
		}
		part speedSensor {
			port speedPort : SpeedSensePort;
		}
		part engine {
			port fuelControlPort : FuelControlPort;
		}
	}
	
	connect driverInterface.setSpeedPort to vehicle.cruiseController.setSpeedPort;
	connect vehicle.speedSensor.speedPort to vehicle.cruiseController.speedPort;
	connect vehicle.cruiseController.fuelControlPort to vehicle.engine.fuelControlPort;
}
