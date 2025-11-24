package 'CruiseControlFunction' {
	private import ScalarValues::Real;
	
	item def SetSpeedCommand {
		attribute targetSpeed : Real;
	}
	
	item def SpeedFeedback {
		attribute actualSpeed : Real;
	}
	
	item def ThrottleControl {
		attribute throttleValue : Real;
	}
	
	port def SetSpeedSource {
		out item command : SetSpeedCommand;
	}
	
	port def SetSpeedSink {
		in item command : SetSpeedCommand;
	}
	
	port def SpeedFeedbackSource {
		out item measurement : SpeedFeedback;
	}
	
	port def SpeedFeedbackSink {
		in item measurement : SpeedFeedback;
	}
	
	port def ThrottleSource {
		out item command : ThrottleControl;
	}
	
	port def ThrottleSink {
		in item command : ThrottleControl;
	}
	
	action def GenerateThrottleControl {
		in setSpeed : SetSpeedCommand;
		in speedFeedback : SpeedFeedback;
		out throttleCommand : ThrottleControl;
	}
	
	part driver {
		port setSpeedPort : SetSpeedSource;
	}
	
	part vehicle {
		part speedometer {
			port speedPort : SpeedFeedbackSource;
		}
		part engine {
			port throttlePort : ThrottleSink;
		}
		part cruiseController {
			port setSpeedPort : SetSpeedSink;
			port speedPort : SpeedFeedbackSink;
			port throttlePort : ThrottleSource;
			perform action generateThrottle : GenerateThrottleControl {
				in setSpeed = setSpeedPort.command;
				in speedFeedback = speedPort.measurement;
				out throttleCommand = throttlePort.command;
			}
		}
	}
	
	connect driver.setSpeedPort to vehicle.cruiseController.setSpeedPort;
	connect vehicle.speedometer.speedPort to vehicle.cruiseController.speedPort;
	connect vehicle.cruiseController.throttlePort to vehicle.engine.throttlePort;
}
