package 'PowertrainControlLogic' {
	private import ScalarValues::*;
	item def FuelCommand;
	action def FuelControlInput {
		out fuelCmd : FuelCommand;
	}
	action def StartEngine {
		in fuelCmd : FuelCommand;
		out engineTorque : Real;
	}
	action def AmplifyTorque {
		in inputTorque : Real;
		out amplifiedTorque : Real;
	}
	action def TransmitTorque {
		in torque : Real;
		out shaftTorque : Real;
	}
	action def DistributeTorque {
		in shaftTorque : Real;
		out wheelTorque : Real;
	}
	action def DetectStall {
		out stallDetected : Boolean;
	}
action def ShutdownPowertrain;
	action def ControlPowertrain;
	action controlPowertrain : ControlPowertrain {
		first start;
		then action command : FuelControlInput {
			out fuelCmd;
		}
		then action startEngine : StartEngine {
			in fuelCmd = command.fuelCmd;
			out engineTorque;
		}
		then action amplify : AmplifyTorque {
			in inputTorque = startEngine.engineTorque;
			out amplifiedTorque;
		}
		then action transmit : TransmitTorque {
			in torque = amplify.amplifiedTorque;
			out shaftTorque;
		}
		then action distribute : DistributeTorque {
			in shaftTorque = transmit.shaftTorque;
			out wheelTorque;
		}
		then action stallMonitor : DetectStall {
			out stallDetected;
		}
		then decide;
			if stallMonitor.stallDetected then shutdown;
	action shutdown : ShutdownPowertrain;
		then done;
	}
}
