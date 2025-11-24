package 'AutomotiveCruiseControl' {
	private import ScalarValues::*;
	
	item def DesiredSpeedRequest {
		attribute targetSpeed : Real;
	}
	
	item def VehicleSpeedFeedback {
		attribute currentSpeed : Real;
	}
	
	item def ThrottleInstruction {
		attribute throttleSetting : Real;
	}
	
	port def DesiredSpeedSource {
		out item request : DesiredSpeedRequest;
	}
	
	port def DesiredSpeedSink {
		in item request : DesiredSpeedRequest;
	}
	
	port def SpeedFeedbackSource {
		out item feedback : VehicleSpeedFeedback;
	}
	
	port def SpeedFeedbackSink {
		in item feedback : VehicleSpeedFeedback;
	}
	
	port def ThrottleCommandSource {
		out item instruction : ThrottleInstruction;
	}
	
	port def ThrottleCommandSink {
		in item instruction : ThrottleInstruction;
	}
	
	action def ComputeThrottleAdjustment {
		in desiredSpeed : DesiredSpeedRequest;
		in speedFeedback : VehicleSpeedFeedback;
		out throttleInstruction : ThrottleInstruction;
	}
	
	part driverInterface {
		port commandOut : DesiredSpeedSource;
	}
	
	part speedometer {
		port speedOut : SpeedFeedbackSource;
	}
	
	part cruiseController {
		port commandIn : DesiredSpeedSink;
		port speedIn : SpeedFeedbackSink;
		port throttleOut : ThrottleCommandSource;
		perform action adjustThrottle : ComputeThrottleAdjustment {
			in desiredSpeed = commandIn.request;
			in speedFeedback = speedIn.feedback;
			out throttleInstruction = throttleOut.instruction;
		}
	}
	
	part engineActuator {
		port throttleIn : ThrottleCommandSink;
	}
	
	connect driverInterface.commandOut to cruiseController.commandIn;
	connect speedometer.speedOut to cruiseController.speedIn;
	connect cruiseController.throttleOut to engineActuator.throttleIn;
	
	requirement def 'AutomaticCruiseRequirement' {
		text = "The cruise control shall transmit the driver’s desired speed, sense actual speed, and adjust engine throttle so the vehicle maintains the preset speed smoothly.";
	}
}
