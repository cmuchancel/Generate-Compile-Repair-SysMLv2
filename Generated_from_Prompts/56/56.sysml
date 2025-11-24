package 'MassAggregationModel' {
	private import ScalarValues::*;
	
	part def Thing {
		attribute simpleMass : Real;
		attribute totalMass : Real;
	}
	
	part def SimpleThing :> Thing {
		attribute redefines totalMass = simpleMass;
	}
	
	part def CompositeThing :> Thing {
		part simpleComponents : SimpleThing[*];
		part compositeComponents : CompositeThing[*];
	}
	
	calc def CompositeMass {
		in composite : CompositeThing;
		return total : Real;
	}
	
	part simpleComponent_1 : SimpleThing {
		attribute simpleMass = 100.0;
	}
	
	part simpleComponent_2 : SimpleThing {
		attribute simpleMass = 150.0;
	}
	
	part compositeRoot : CompositeThing {
		attribute simpleMass = 200.0;
		part simpleComponents {
			part frontAssembly : SimpleThing = simpleComponent_1;
			part rearAssembly : SimpleThing = simpleComponent_2;
		}
		attribute redefines totalMass = simpleMass + simpleComponents.frontAssembly.totalMass + simpleComponents.rearAssembly.totalMass;
	}
	
	calc compositeMassRollup : CompositeMass {
		in composite = compositeRoot;
		return total;
	}
}
