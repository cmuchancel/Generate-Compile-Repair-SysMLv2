package 'MassThresholdAggregation' {
	private import ScalarValues::*;
	
	part def Component {
		attribute basicMass : Real;
		attribute totalMass : Real;
		part subcomponents : Component[*];
		attribute filterThreshold : Real;
	}
	
	calc def FilteredMass {
		in component : Component;
		return total : Real;
	}
	
	part componentRoot : Component {
		attribute basicMass = 200.0;
		attribute filterThreshold = 100.0;
		part subcomponents {
			part majorSubassembly : Component {
				attribute basicMass = 150.0;
				part subcomponents {
					part minorComponent {
						attribute basicMass = 80.0;
					}
				}
			}
			part heavyComponent : Component {
				attribute basicMass = 180.0;
			}
		}
		attribute redefines totalMass = basicMass + subcomponents.heavyComponent.basicMass + subcomponents.majorSubassembly.basicMass;
	}
	
	calc filteredMass : FilteredMass {
		in component = componentRoot;
		return total;
	}
}
