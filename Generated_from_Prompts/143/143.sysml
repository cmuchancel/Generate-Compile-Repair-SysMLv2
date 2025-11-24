package 'CoreDataTypeManagement' {
	private import ScalarValues::*;
	private import Time::*;
	private import SI::*;
	abstract item def ManagedDataType {
		attribute typeName : String;
	}
	item def UnsignedIntegerData :> ManagedDataType {
		attribute typeName = "UnsignedInteger";
		attribute value : Integer;
		constraint nonNegative {
			constraint { value >= 0 }
		}
	}
	item def UnsignedFloatData :> ManagedDataType {
		attribute typeName = "UnsignedFloat";
		attribute value : Real;
		constraint nonNegative {
			constraint { value >= 0.0 }
		}
	}
	enum def StatusLevel {
		enum Critical;
		enum Warning;
		enum Normal;
	}
	enum def StatusColor {
		enum Red;
		enum Yellow;
		enum Green;
	}
	part def StatusCondition {
		attribute level : StatusLevel;
		attribute color : StatusColor;
	}
	enum def DiameterOption {
		enum Small;
		enum Medium;
		enum Large;
	}
	part def DiameterParameter {
		attribute selectedOption : DiameterOption;
		attribute diameter : LengthValue;
	}
	part def SmallDiameterSetting :> DiameterParameter {
		attribute redefines selectedOption = DiameterOption::Small;
		attribute redefines diameter = 60 [mm];
	}
	part def MediumDiameterSetting :> DiameterParameter {
		attribute redefines selectedOption = DiameterOption::Medium;
		attribute redefines diameter = 70 [mm];
	}
	part def LargeDiameterSetting :> DiameterParameter {
		attribute redefines selectedOption = DiameterOption::Large;
		attribute redefines diameter = 80 [mm];
	}
	part def DataTypeProfile {
		part integerData : UnsignedIntegerData;
		part floatData : UnsignedFloatData;
		part status : StatusCondition;
		part diameterSetting : SmallDiameterSetting;
		attribute recordedAt : Time::DateTime;
		attribute statusMessage : String;
		attribute acknowledged : Boolean;
	}
	part statusPalette {
		part criticalStatus : StatusCondition {
			attribute redefines level = StatusLevel::Critical;
			attribute redefines color = StatusColor::Red;
		}
		part warningStatus : StatusCondition {
			attribute redefines level = StatusLevel::Warning;
			attribute redefines color = StatusColor::Yellow;
		}
		part normalStatus : StatusCondition {
			attribute redefines level = StatusLevel::Normal;
			attribute redefines color = StatusColor::Green;
		}
	}
}
