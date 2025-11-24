package 'VehicleMassAggregationSystem' {
	private import ScalarValues::*;
	part def Component {
		attribute identifier : String;
		attribute ownMassKilograms : Real;
		attribute totalMassKilograms : Real;
	}
	part def BodyStructure :> Component {
		part structuralFrame : Component;
		part exteriorPanels : Component;
		part interiorSystems : Component;
		assert constraint bodyMassAggregation {
			constraint { totalMassKilograms = ownMassKilograms + structuralFrame.totalMassKilograms + exteriorPanels.totalMassKilograms + interiorSystems.totalMassKilograms; }
		}
	}
	part def VehicleAssembly :> Component {
		part bodyStructure : BodyStructure;
		part engineModule : Component;
		part transmissionModule : Component;
		part wheelSet : Component;
		assert constraint vehicleMassAggregation {
			constraint {
				totalMassKilograms = ownMassKilograms
					+ bodyStructure.totalMassKilograms
					+ engineModule.totalMassKilograms
					+ transmissionModule.totalMassKilograms
					+ wheelSet.totalMassKilograms;
			}
		}
	}
	part vehicleMassHierarchy : VehicleAssembly {
		attribute identifier = "Vehicle-001";
		attribute ownMassKilograms = 600.0;
		attribute totalMassKilograms = 2280.0;
		part bodyStructure : BodyStructure redefines bodyStructure {
			attribute identifier = "BodyStructure";
			attribute ownMassKilograms = 500.0;
			attribute totalMassKilograms = 970.0;
			part structuralFrame : Component redefines structuralFrame {
				attribute identifier = "StructuralFrame";
				attribute ownMassKilograms = 200.0;
				attribute totalMassKilograms = ownMassKilograms;
			}
			part exteriorPanels : Component redefines exteriorPanels {
				attribute identifier = "ExteriorPanels";
				attribute ownMassKilograms = 150.0;
				attribute totalMassKilograms = ownMassKilograms;
			}
			part interiorSystems : Component redefines interiorSystems {
				attribute identifier = "InteriorSystems";
				attribute ownMassKilograms = 120.0;
				attribute totalMassKilograms = ownMassKilograms;
			}
		}
		part engineModule : Component redefines engineModule {
			attribute identifier = "Engine";
			attribute ownMassKilograms = 320.0;
			attribute totalMassKilograms = ownMassKilograms;
		}
		part transmissionModule : Component redefines transmissionModule {
			attribute identifier = "Transmission";
			attribute ownMassKilograms = 210.0;
			attribute totalMassKilograms = ownMassKilograms;
		}
		part wheelSet : Component redefines wheelSet {
			attribute identifier = "WheelSet";
			attribute ownMassKilograms = 180.0;
			attribute totalMassKilograms = ownMassKilograms;
		}
	}
	requirement def 'TransparentMassAggregationRequirement' {
		text = "Each component shall record its identifier, own mass, and aggregated mass, supporting automated top-down computation for vehicles and constituent assemblies.";
	}
}
