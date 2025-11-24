package 'AutomotiveECUCommunication' {
	port def BusIF;
	port def SensorIF;
	connection def CANLink {
		end controllerPort : BusIF;
		end busPort : BusIF;
	}
	connection def SensorLink {
		end controllerPort : SensorIF;
		end sensorPort : SensorIF;
	}
	part def CANBus {
		port vcuPort : BusIF;
		port ecuPort : BusIF;
	}
	part def VehicleControlUnit;
	part def EngineControlUnit;
	part def SensorGateway;
	part vehicleControlUnit : VehicleControlUnit {
		port busInterface : BusIF;
		port sensorInterface : SensorIF;
	}
	part engineControlUnit : EngineControlUnit {
		port busInterface : BusIF;
	}
	part canBus : CANBus;
	part sensorGateway : SensorGateway {
		port sensorPort : SensorIF;
	}
	connection vcuCanConnection : CANLink {
		end controllerPort = vehicleControlUnit.busInterface;
		end busPort = canBus.vcuPort;
	}
	connection ecuCanConnection : CANLink {
		end controllerPort = engineControlUnit.busInterface;
		end busPort = canBus.ecuPort;
	}
	connection vcuSensorConnection : SensorLink {
		end controllerPort = vehicleControlUnit.sensorInterface;
		end sensorPort = sensorGateway.sensorPort;
	}
}
