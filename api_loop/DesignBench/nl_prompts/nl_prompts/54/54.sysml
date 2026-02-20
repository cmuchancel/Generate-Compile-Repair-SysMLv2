package 'VehicleMassAggregation' {
	private import ScalarValues::*;
	private import NumericalFunctions::*;
	part def MassTrackedComponent {
		attribute identifier : String;
		attribute ownMass :> ISQ::mass;
		part subcomponents : MassTrackedComponent[*];
		attribute totalMass :> ISQ::mass default ownMass + sum(subcomponents.totalMass);
	}
	part def SimpleComponent :> MassTrackedComponent;
	part def CompositeComponent :> MassTrackedComponent;
	part def CoreComponent :> SimpleComponent;
	part def Vehicle :> CompositeComponent {
		part body : CompositeComponent;
		part engine : CoreComponent;
		part transmission : CoreComponent;
		part otherComponents : MassTrackedComponent[*];
	}
	part def VehicleMassSystem {
		part vehicles : Vehicle[*];
		ref part queriedVehicle : Vehicle;
		attribute queryResult :> ISQ::mass default queriedVehicle.totalMass;
	}
}
