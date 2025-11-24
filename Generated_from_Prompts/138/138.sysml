package 'PowertrainConfigurationRules' {
	private import ScalarValues::*;
	enum def EngineVariant {
		enum FourCylinder;
		enum SixCylinder;
	}
	enum def TransmissionVariant {
		enum Manual;
		enum Automatic;
	}
	part def PowertrainConfiguration {
		attribute isHighPerformance : Boolean;
		attribute engineVariant : EngineVariant;
		attribute transmissionVariant : TransmissionVariant;
	}
	constraint def HighPerformanceConstraint {
		in isHighPerformance : Boolean;
		in engineVariant : EngineVariant;
		assert constraint {
			(isHighPerformance implies engineVariant = EngineVariant::SixCylinder)
				and (not isHighPerformance implies engineVariant = EngineVariant::FourCylinder)
		}
	}
	constraint def AllowedCombinations {
		in engineVariant : EngineVariant;
		in transmissionVariant : TransmissionVariant;
		assert constraint {
			(engineVariant = EngineVariant::FourCylinder and transmissionVariant = TransmissionVariant::Manual)
				or (engineVariant = EngineVariant::SixCylinder and transmissionVariant = TransmissionVariant::Automatic)
		}
	}
	part powertrainRules {
		part configuration : PowertrainConfiguration;
		part highPerformanceVehicle : PowertrainConfiguration {
			attribute isHighPerformance = true;
			attribute engineVariant = EngineVariant::SixCylinder;
			attribute transmissionVariant = TransmissionVariant::Automatic;
		}
		part standardVehicle : PowertrainConfiguration {
			attribute isHighPerformance = false;
			attribute engineVariant = EngineVariant::FourCylinder;
			attribute transmissionVariant = TransmissionVariant::Manual;
		}
		part highPerformanceConstraint : HighPerformanceConstraint {
			in isHighPerformance = highPerformanceVehicle.isHighPerformance;
			in engineVariant = highPerformanceVehicle.engineVariant;
		}
		part standardPerformanceConstraint : HighPerformanceConstraint {
			in isHighPerformance = standardVehicle.isHighPerformance;
			in engineVariant = standardVehicle.engineVariant;
		}
		part highPerformanceCombination : AllowedCombinations {
			in engineVariant = highPerformanceVehicle.engineVariant;
			in transmissionVariant = highPerformanceVehicle.transmissionVariant;
		}
		part standardCombination : AllowedCombinations {
			in engineVariant = standardVehicle.engineVariant;
			in transmissionVariant = standardVehicle.transmissionVariant;
		}
	}
}
