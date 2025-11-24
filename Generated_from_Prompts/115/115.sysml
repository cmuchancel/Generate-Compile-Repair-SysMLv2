package 'AutomotiveConfigurationManagement' {
	private import ScalarValues::*;
	part def Component1;
	part def Component2;
	part def Component3;
	part def Component4;
	part def Component5;
	part def Component6;
	part def SubsystemAChoice {
		attribute isSecondConfiguration : Boolean;
	}
	part def SubsystemBChoice {
		attribute isThirdConfiguration : Boolean;
	}
	part def SubsystemAOption1 :> SubsystemAChoice {
		part component1 : Component1;
		part component2 : Component2;
		attribute redefines isSecondConfiguration = false;
	}
	part def SubsystemAOption2 :> SubsystemAChoice {
		part component2 : Component2;
		part component3 : Component3;
		attribute redefines isSecondConfiguration = true;
	}
	part def SubsystemBOption1 :> SubsystemBChoice {
		part component4 : Component4;
		part component5 : Component5;
		attribute redefines isThirdConfiguration = true;
	}
	part def SubsystemBOption2 :> SubsystemBChoice {
		part component5 : Component5;
		part component6 : Component6;
		attribute redefines isThirdConfiguration = false;
	}
	constraint def ConfigurationCompatibility {
		in subsystemA : SubsystemAChoice;
		in subsystemB : SubsystemBChoice;
		assert constraint { not subsystemA.isSecondConfiguration or subsystemB.isThirdConfiguration };
	}
	part def VehicleConfiguration {
		part subsystemA : SubsystemAChoice;
		part subsystemB : SubsystemBChoice;
		constraint compatibility : ConfigurationCompatibility {
			in subsystemA = VehicleConfiguration::subsystemA;
			in subsystemB = VehicleConfiguration::subsystemB;
		}
	}
	part configurationA : VehicleConfiguration {
		part subsystemA : SubsystemAOption1 redefines subsystemA;
		part subsystemB : SubsystemBOption1 redefines subsystemB;
	}
	part configurationB : VehicleConfiguration {
		part subsystemA : SubsystemAOption2 redefines subsystemA;
		part subsystemB : SubsystemBOption1 redefines subsystemB;
	}
}
