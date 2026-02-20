package 'MassFilteringAggregation' {
	private import ScalarValues::*;
	private import NumericalFunctions::*;
	part def MassBearingObject {
		attribute basicMass :> ISQ::mass;
		part subcomponents : MassBearingObject[*];
		attribute minCountedMass :> ISQ::mass;
		attribute totalMass :> ISQ::mass default basicMass + sum(subcomponents.totalMass);
		attribute filteredTotalMass :> ISQ::mass default basicMass + sum(subcomponents.totalMass.?{in m:>ISQ::mass; m >= minCountedMass});
	}
	part def SimpleMassObject :> MassBearingObject {
		part subcomponents : MassBearingObject[0];
	}
	part def CompositeMassObject :> MassBearingObject;
	part def MassAggregationController {
		part managedObjects : MassBearingObject[*];
	}
}
