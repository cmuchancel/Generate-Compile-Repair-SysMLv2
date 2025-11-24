package 'VehicleFuelEconomy073' {
	private import ScalarValues::*;
	
	part def VehicleParameters {
		attribute curbWeight : Real;
		attribute load : Real;
		attribute wheelDiameter : Real;
		attribute transmissionEfficiency : Real;
	}
	
part def DrivingScenario {
	attribute name : String;
	attribute averageSpeed : Real;
	attribute distanceMiles : Real;
	attribute fuelUsedGallons : Real;
}

part def DrivePoint {
	attribute timestamp : Real;
	attribute position : Real;
	attribute speed : Real;
}

part def DrivingCycle {
	part point : DrivePoint[*] ordered;
}

part def DriveSegment {
	attribute initialSpeed : Real;
	attribute finalSpeed : Real;
	attribute timeInterval : Real;
	attribute distanceMiles : Real;
	attribute fuelGallonsUsed : Real;
}
	
	calc def ComputeAcceleration {
		in initialSpeed : Real;
		in finalSpeed : Real;
		in timeInterval : Real;
		return acceleration : Real = (finalSpeed - initialSpeed) / timeInterval;
	}
	
	calc def ComputeWheelPower {
		in vehicleMass : Real;
		in acceleration : Real;
		in speed : Real;
		in transmissionEfficiency : Real;
		return wheelPower : Real = (vehicleMass * acceleration * speed) / transmissionEfficiency;
	}
	
	calc def ComputeFuelEconomyMpg {
		in distanceMiles : Real;
		in fuelGallons : Real;
		return mpg : Real = distanceMiles / fuelGallons;
	}
	
	calc def ConvertMpgToLitersPer100km {
		in mpg : Real;
		return litersPer100km : Real = 235.214583 / mpg;
	}
	
	part def PerformanceAnalysis {
		part vehicle : VehicleParameters;
		part drivingCycle : DrivingCycle;
		part driveSegment : DriveSegment;
		part urbanScenario : DrivingScenario;
		part highwayScenario : DrivingScenario;
		attribute targetFuelEconomyMpg : Real = 30.0;
		attribute vehicleMass : Real = vehicle.curbWeight + vehicle.load;
		calc urbanFuelEconomy : ComputeFuelEconomyMpg {
			in distanceMiles = urbanScenario.distanceMiles;
			in fuelGallons = urbanScenario.fuelUsedGallons;
			return mpg;
		}
		calc highwayFuelEconomy : ComputeFuelEconomyMpg {
			in distanceMiles = highwayScenario.distanceMiles;
			in fuelGallons = highwayScenario.fuelUsedGallons;
			return mpg;
		}
		calc urbanLitersPer100km : ConvertMpgToLitersPer100km {
			in mpg = urbanFuelEconomy.mpg;
			return litersPer100km;
		}
		calc highwayLitersPer100km : ConvertMpgToLitersPer100km {
			in mpg = highwayFuelEconomy.mpg;
			return litersPer100km;
		}
		attribute meetsFuelEconomyTarget : Boolean = urbanFuelEconomy.mpg > targetFuelEconomyMpg and highwayFuelEconomy.mpg > targetFuelEconomyMpg;
	}

}
