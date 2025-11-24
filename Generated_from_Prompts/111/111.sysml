package 'AutomobileAssemblyProcess' {
	part def Gearbox;
	part def Engine;
	part def VehicleAssembly;
	part def VehicleWithGearbox :> VehicleAssembly {
		part gearbox : Gearbox;
	}
	part def CompleteVehicle :> VehicleWithGearbox {
		part engine : Engine;
	}
	action def InstallGearbox {
		in baseVehicle : VehicleAssembly;
		in gearbox : Gearbox;
		out assembledVehicle : VehicleWithGearbox;
	}
	action def InstallEngine {
		in vehicleWithGearbox : VehicleWithGearbox;
		in engine : Engine;
		out finishedVehicle : CompleteVehicle;
	}
	part finalAssemblyLine {
		part baseVehicle : VehicleAssembly;
		part gearboxUnit : Gearbox;
		part engineUnit : Engine;
		part vehicleAfterGearbox : VehicleWithGearbox;
		part completedVehicle : CompleteVehicle;
		action assemblySequence {
			first start;
			then action installGearbox : InstallGearbox {
				in baseVehicle = finalAssemblyLine::baseVehicle;
				in gearbox = finalAssemblyLine::gearboxUnit;
				out assembledVehicle = finalAssemblyLine::vehicleAfterGearbox;
			}
			then action installEngine : InstallEngine {
				in vehicleWithGearbox = finalAssemblyLine::vehicleAfterGearbox;
				in engine = finalAssemblyLine::engineUnit;
				out finishedVehicle = finalAssemblyLine::completedVehicle;
			}
			then done;
		}
	}
}
