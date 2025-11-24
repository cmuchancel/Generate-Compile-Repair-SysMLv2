package 'VehiclePowerTransmission' {
	private import ScalarValues::*;
	item def FuelCommand;
	port def FuelCommandPort {
		in item command : FuelCommand;
	}
	port def PowerPort;
	port def ClutchPort;
	port def ShaftPort;
	port def WheelRoadPort;
part def Engine {
	port fuelCmdPort : FuelCommandPort;
		port powerOutput : PowerPort;
		port clutchControl : ClutchPort;
	}
	part def Transmission {
		port powerInput : PowerPort;
		port powerOutput : PowerPort;
		port clutchControl : ClutchPort;
	}
	part def Clutch {
		port enginePower : PowerPort;
		port transmissionPower : PowerPort;
		port engineControl : ClutchPort;
		port transmissionControl : ClutchPort;
	}
	part def DriveShaft {
		port inputPower : PowerPort;
		port outputPower : PowerPort;
	}
	part def Differential {
		port inputPower : PowerPort;
		port leftHalfShaft : ShaftPort;
		port rightHalfShaft : ShaftPort;
	}
	part def HalfShaft {
		port differentialInterface : ShaftPort;
		port wheelInterface : ShaftPort;
	}
	part def Wheel {
		port axleInterface : ShaftPort;
		port roadInterface : WheelRoadPort;
	}
part def RoadSurface {
	port leftContact : WheelRoadPort;
	port rightContact : WheelRoadPort;
}
part def FuelController {
	port commandOut : FuelCommandPort;
}
	part def RearAxleAssembly {
		part differential : Differential;
		part leftHalfShaft : HalfShaft;
		part rightHalfShaft : HalfShaft;
		part leftWheel : Wheel;
		part rightWheel : Wheel;
	}
	part powerTransmissionSystem {
		part engine : Engine;
		part fuelController : FuelController;
		part clutch : Clutch;
		part transmission : Transmission;
		part driveShaft : DriveShaft;
		part rearAxle : RearAxleAssembly;
		part roadInterface : RoadSurface;
		bind engine.powerOutput = clutch.enginePower;
		bind fuelController.commandOut = engine.fuelCmdPort;
		bind clutch.transmissionPower = transmission.powerInput;
		bind transmission.powerOutput = driveShaft.inputPower;
		bind driveShaft.outputPower = rearAxle.differential.inputPower;
		bind rearAxle.differential.leftHalfShaft = rearAxle.leftHalfShaft.differentialInterface;
		bind rearAxle.differential.rightHalfShaft = rearAxle.rightHalfShaft.differentialInterface;
		bind rearAxle.leftHalfShaft.wheelInterface = rearAxle.leftWheel.axleInterface;
		bind rearAxle.rightHalfShaft.wheelInterface = rearAxle.rightWheel.axleInterface;
		bind engine.clutchControl = clutch.engineControl;
		bind transmission.clutchControl = clutch.transmissionControl;
		bind rearAxle.leftWheel.roadInterface = roadInterface.leftContact;
		bind rearAxle.rightWheel.roadInterface = roadInterface.rightContact;
	}
}
