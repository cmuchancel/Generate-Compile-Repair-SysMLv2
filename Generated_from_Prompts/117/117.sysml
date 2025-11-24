package 'VehicleProductLineConfiguration' {
	private import ScalarValues::*;
	enum def PerformanceLevel {
		enum High;
		enum Standard;
	}
	part def Engine {
		attribute cylinders : Integer;
	}
	part def SixCylinderEngine :> Engine {
		attribute redefines cylinders = 6;
	}
	part def FourCylinderEngine :> Engine {
		attribute redefines cylinders = 4;
	}
	part def Transmission {
		attribute mode : String;
	}
	part def AutomaticTransmission :> Transmission {
		attribute redefines mode = "Automatic";
	}
	part def ManualTransmission :> Transmission {
		attribute redefines mode = "Manual";
	}
	part def WheelSet {
		attribute rimProfile : String;
	}
	part def WideRimWheel :> WheelSet {
		attribute redefines rimProfile = "Wide";
	}
	part def NarrowRimWheel :> WheelSet {
		attribute redefines rimProfile = "Narrow";
	}
	part def VehicleConfiguration {
		attribute performanceLevel : PerformanceLevel;
		part engine : Engine;
		part transmission : Transmission;
		part wheels : WheelSet;
		assert constraint alignment : PerformanceCompatibility {
			in performance = VehicleConfiguration::performanceLevel;
			in engine = VehicleConfiguration::engine;
			in transmission = VehicleConfiguration::transmission;
			in wheels = VehicleConfiguration::wheels;
		}
	}
	constraint def PerformanceCompatibility {
		in performance : PerformanceLevel;
		in engine : Engine;
		in transmission : Transmission;
		in wheels : WheelSet;
		(performance == PerformanceLevel::High
			and engine istype SixCylinderEngine
			and transmission istype AutomaticTransmission
			and wheels istype WideRimWheel)
		or
		(performance == PerformanceLevel::Standard
			and engine istype FourCylinderEngine
			and transmission istype ManualTransmission
			and wheels istype NarrowRimWheel)
	}
	part def VehicleModelVariant {
		attribute name : String;
		part configuration : VehicleConfiguration;
	}
	part productLinePortfolio {
		part highPerformanceVariant : VehicleModelVariant {
			attribute name = "HighPerformance";
			part configuration : VehicleConfiguration redefines configuration {
				attribute redefines performanceLevel = PerformanceLevel::High;
				part engine : SixCylinderEngine redefines engine;
				part transmission : AutomaticTransmission redefines transmission;
				part wheels : WideRimWheel redefines wheels;
			}
		}
		part standardVariant : VehicleModelVariant {
			attribute name = "StandardPerformance";
			part configuration : VehicleConfiguration redefines configuration {
				attribute redefines performanceLevel = PerformanceLevel::Standard;
				part engine : FourCylinderEngine redefines engine;
				part transmission : ManualTransmission redefines transmission;
				part wheels : NarrowRimWheel redefines wheels;
			}
		}
	}
	action def MatchConfiguration {
		in desiredPerformance : PerformanceLevel;
		out matchedConfiguration : VehicleConfiguration;
	}
	action highPerformanceMatch : MatchConfiguration {
		in desiredPerformance = PerformanceLevel::High;
		out matchedConfiguration = productLinePortfolio.highPerformanceVariant.configuration;
	}
	action standardPerformanceMatch : MatchConfiguration {
		in desiredPerformance = PerformanceLevel::Standard;
		out matchedConfiguration = productLinePortfolio.standardVariant.configuration;
	}
	requirement def 'FlexibleProductLineRequirement' {
		text = "Provide automatic matching of engines, transmissions, and wheels for high and standard performance levels while preventing incompatible combinations.";
	}
	requirement def 'VariantManagementRequirement' {
		text = "Support multiple vehicle model variants so enterprises can expand product lines based on user preferences and market demand.";
	}
}
