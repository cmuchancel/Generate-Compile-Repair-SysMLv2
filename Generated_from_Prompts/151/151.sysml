package 'UserTransportationService' {
	private import ScalarValues::*;
	
	part def Driver;
	part def Passenger;
	
	action def BoardVehicle {
		in driver : Driver;
		in passengers : Passenger[0..4];
	}
	
	action def OperateVehicle {
		in driver : Driver;
	}
	
	action def EvaluateFuelLevel {
		in fuelPercentage : Real;
	}
	
	action def RefuelVehicle {
		in refuelCommand : Boolean;
		in driver : Driver;
	}
	
	action def ExitVehicle {
		in driver : Driver;
		in passengers : Passenger[0..4];
	}
	
	part def TransportVehicle {
		attribute maxFuelCapacityLiters : Real;
		attribute currentFuelLiters : Real;
		attribute fuelLevelPercent : Real = (currentFuelLiters / maxFuelCapacityLiters) * 100.0;
		attribute refuelThresholdPercent : Real = 10.0;
		attribute needsRefuel : Boolean = fuelLevelPercent < refuelThresholdPercent;
		attribute environmentFactorsConsidered : Boolean = true;
		part driverSeat : Driver;
		part passengers : Passenger[0..4];
		perform action boardOccupants : BoardVehicle {
			in driver = driverSeat;
			in passengers = passengers;
		}
		perform action operateJourney : OperateVehicle {
			in driver = driverSeat;
		}
		perform action monitorFuel : EvaluateFuelLevel {
			in fuelPercentage = fuelLevelPercent;
		}
		perform action refuelAtStation : RefuelVehicle {
			in refuelCommand = needsRefuel;
			in driver = driverSeat;
		}
		perform action exitOccupants : ExitVehicle {
			in driver = driverSeat;
			in passengers = passengers;
		}
	}
	
	part transportationService : TransportVehicle {
		attribute maxFuelCapacityLiters = 50.0;
		attribute currentFuelLiters = 4.0;
	}
	
	requirement def 'TransportationEfficiencyRequirement' {
		text = "The transportation service shall board a driver with up to four passengers, operate the trip, and ensure everyone exits safely.";
	}
	
	requirement def 'FuelContinuityRequirement' {
		text = "When fuel level drops below ten percent of maximum capacity the driver shall refuel at a gas station to keep the vehicle operating.";
	}
	
	requirement def 'EnvironmentalAwarenessRequirement' {
		text = "Environmental factors must be considered throughout every transportation task to preserve safe and efficient travel.";
	}
}
