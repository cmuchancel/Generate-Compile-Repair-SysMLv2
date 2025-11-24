package 'CruiseControlSystem' {
	private import ScalarValues::Real;
	
	item def SetSpeedCommand {
		attribute targetSpeed : Real;
	}
	
	item def SpeedMeasurement {
		attribute actualSpeed : Real;
	}
	
	item def FuelControlCommand {
		attribute throttleSetting : Real;
	}
	
	port def SetSpeedSource {
		out item command : SetSpeedCommand;
	}
	
	port def SetSpeedSink {
		in item command : SetSpeedCommand;
	}
	
	port def SpeedMeasurementSource {
		out item measurement : SpeedMeasurement;
	}
	
	port def SpeedMeasurementSink {
		in item measurement : SpeedMeasurement;
	}
	
	port def FuelControlSource {
		out item command : FuelControlCommand;
	}
	
	port def FuelControlSink {
		in item command : FuelControlCommand;
	}
	
	part driver {
		port setSpeedPort : SetSpeedSource;
	}
	
part vehicle {
		part cruiseController {
			port setSpeedPort : SetSpeedSink;
			port speedPort : SpeedMeasurementSink;
			port fuelCommandPort : FuelControlSource;
		}
		part speedometer {
			port speedPort : SpeedMeasurementSource;
			attribute sensedSpeed : Real;
		}
		part engine {
			port fuelCommandPort : FuelControlSink;
		}
	}
	
connect driver.setSpeedPort to vehicle.cruiseController.setSpeedPort;
connect vehicle.speedometer.speedPort to vehicle.cruiseController.speedPort;
connect vehicle.cruiseController.fuelCommandPort to vehicle.engine.fuelCommandPort;
}
