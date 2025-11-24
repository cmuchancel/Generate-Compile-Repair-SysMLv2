package 'PowertrainIntegratedManagement' {
	private import ScalarValues::*;
	item def FuelCommand;
	action def FuelInput {
		out fuelCmd : FuelCommand;
	}
	action def EngineStartSignalInput {
		out startActivated : Boolean;
	}
	action def EngineShutdownSignalInput {
		out shutdownActivated : Boolean;
	}
	action def GenerateEngineTorque {
		in fuelCmd : FuelCommand;
		out engineTorque : Real;
	}
	action def AmplifyTransmissionTorque {
		in inputTorque : Real;
		out outputTorque : Real;
	}
	action def TransmitThroughDriveShaft {
		in torque : Real;
		out shaftTorque : Real;
	}
action def DistributeWheelPower {
	in shaftTorque : Real;
	out leftWheelTorque : Real;
	out rightWheelTorque : Real;
}
	action def TerminatePowerOutput;
	action def ManagePowertrain;
	action managePowertrain : ManagePowertrain {
		first start;
		then action fuel : FuelInput {
			out fuelCmd;
		}
		then action startSignal : EngineStartSignalInput {
			out startActivated;
		}
		then decide;
			if startSignal.startActivated then torqueFlow;
		action torqueFlow {
			action engineTorque : GenerateEngineTorque {
				in fuelCmd = fuel.fuelCmd;
				out engineTorque;
			}
			action amplify : AmplifyTransmissionTorque {
				in inputTorque = engineTorque.engineTorque;
				out outputTorque;
			}
			action transmit : TransmitThroughDriveShaft {
				in torque = amplify.outputTorque;
				out shaftTorque;
			}
			action distribute : DistributeWheelPower {
				in shaftTorque = transmit.shaftTorque;
				out leftWheelTorque;
				out rightWheelTorque;
			}
		}
		then action shutdownSignal : EngineShutdownSignalInput {
			out shutdownActivated;
		}
		then decide;
			if shutdownSignal.shutdownActivated then stopOutput;
		action stopOutput : TerminatePowerOutput;
		then done;
	}
}
