package CoreDataTypes {
    private import ScalarValues::*;

    enum def StatusColor {
        enum red;
        enum yellow;
        enum green;
    }

    enum def DiameterSize {
        enum small = "Small (60 mm)";
        enum medium = "Medium (70 mm)";
        enum large = "Large (80 mm)";
    }

    part def UnsignedInteger {
        attribute value : Integer;
        constraint {
            value >= 0
        }
    }

    part def UnsignedFloat {
        attribute value : Real;
        constraint {
            value >= 0.0
        }
    }

    part def StatusCondition {
        attribute level : StatusColor;
        attribute color : StatusColor = level;
    }

    part def PhysicalDimension {
        attribute diameter : DiameterSize = DiameterSize::small;
    }

    part def CoreDataManager {
        attribute unsignedInt : UnsignedInteger;
        attribute unsignedFloat : UnsignedFloat;
        attribute dateTime : DateTime;
        attribute stringValue : String;
        attribute booleanValue : Boolean;
        attribute status : StatusCondition;
        attribute dimension : PhysicalDimension;
    }
}