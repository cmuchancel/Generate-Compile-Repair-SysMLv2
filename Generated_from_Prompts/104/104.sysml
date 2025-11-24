package 'PowertrainFunctionalArchitecture' {
	private import ScalarValues::*;
	item def FuelCommand;
	action def InputFuelCommand {
		out fuelCmd : FuelCommand;
	}
	action def ShutdownCommandInput {
		out shutdownRequested : Boolean;
	}
	action def StartEngine {
		in fuelCmd : FuelCommand;
		out engineTorque : Real;
	}
	action def AmplifyTransmission {
		in inputTorque : Real;
		out outputTorque : Real;
	}
	action def TransmitDriveShaft {
		in torque : Real;
		out shaftTorque : Real;
	}
	action def DistributeWheelTorque {
		in shaftTorque : Real;
		out leftWheelTorque : Real;
		out rightWheelTorque : Real;
	}
	action def ShutdownPowertrain;
	action def ControlPowertrain;
	action controlPowertrain : ControlPowertrain {
		first start;
		then action fuelInput : InputFuelCommand {
			out fuelCmd;
		}
		then action shutdownRequest : ShutdownCommandInput {
			out shutdownRequested;
		}
		then decide;
			if not shutdownRequest.shutdownRequested then torqueFlow;
			if shutdownRequest.shutdownRequested then shutdown;
		action torqueFlow {
			action startEngine : StartEngine {
				in fuelCmd = fuelInput.fuelCmd;
				out engineTorque;
			}
			action amplify : AmplifyTransmission {
				in inputTorque = startEngine.engineTorque;
				out outputTorque;
			}
			action driveShaft : TransmitDriveShaft {
				in torque = amplify.outputTorque;
				out shaftTorque;
			}
			action distribute : DistributeWheelTorque {
				in shaftTorque = driveShaft.shaftTorque;
				out leftWheelTorque;
				out rightWheelTorque;
			}
		}
		action shutdown : ShutdownPowertrain;
		then done;
	}
}
