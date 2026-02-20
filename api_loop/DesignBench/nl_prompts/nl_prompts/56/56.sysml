package 'MassAggregationSystem' {
	private import ScalarValues::*;
	private import NumericalFunctions::*;
	part def MassObject {
		attribute simpleMass :> ISQ::mass;
		attribute totalMass :> ISQ::mass;
	}
	part def SimpleThing :> MassObject {
		attribute :>> totalMass = simpleMass;
	}
	part def CompositeThing :> MassObject {
		part subcomponents : MassObject[*];
		attribute :>> totalMass = simpleMass + sum(subcomponents.totalMass);
	}
	part def MassAggregationRegistry {
		part registeredThings : MassObject[*];
	}
}
