package DataManagementSystem {
    private import ScalarValues::*;

    enum def StatusColor {
        enum red;
        enum yellow;
        enum green;
    }

    enum def SeverityLevel {
        enum critical;
        enum warning;
        enum normal;
    }

    enum def DiameterOption {
        enum Small;
        enum Medium;
        enum Large;
    }

    part def UnsignedIntegerType {
        attribute value : Integer;
        attribute isNonNegative : Boolean = true;
    }

    part def UnsignedFloatType {
        attribute value : Real;
        attribute isNonNegative : Boolean = true;
    }

    part def DateTimeType {
        attribute dateTimeValue : String;
    }

    part def StringType {
        attribute stringValue : String;
    }

    part def BooleanType {
        attribute booleanValue : Boolean;
    }

    part def StatusCondition {
        attribute severity : SeverityLevel;
        attribute color : StatusColor;
        attribute criticalColor : StatusColor = StatusColor::red;
        attribute warningColor : StatusColor = StatusColor::yellow;
        attribute normalColor : StatusColor = StatusColor::green;
    }

    part def PhysicalDimension {
        attribute diameterOption : DiameterOption = DiameterOption::Small;
        attribute smallDiameter : Real = 60.0;
        attribute mediumDiameter : Real = 70.0;
        attribute largeDiameter : Real = 80.0;
        attribute diameterUnit : String = "mm";
    }

    part def DataManagementCore {
        part uintData : UnsignedIntegerType;
        part ufloatData : UnsignedFloatType;
        part dateTimeData : DateTimeType;
        part stringData : StringType;
        part boolData : BooleanType;
        part statusData : StatusCondition;
        part dimensionData : PhysicalDimension;
    }

    part system : DataManagementCore {
        part uintData : UnsignedIntegerType {
            attribute value : Integer = 0;
            attribute isNonNegative : Boolean = true;
        }
        part ufloatData : UnsignedFloatType {
            attribute value : Real = 0.0;
            attribute isNonNegative : Boolean = true;
        }
        part dateTimeData : DateTimeType {
            attribute dateTimeValue : String = "1970-01-01T00:00:00Z";
        }
        part stringData : StringType {
            attribute stringValue : String = "";
        }
        part boolData : BooleanType {
            attribute booleanValue : Boolean = false;
        }
        part statusData : StatusCondition {
            attribute severity : SeverityLevel = SeverityLevel::normal;
            attribute color : StatusColor = StatusColor::green;
        }
        part dimensionData : PhysicalDimension {
            attribute diameterOption : DiameterOption = DiameterOption::Small;
        }
    }
}