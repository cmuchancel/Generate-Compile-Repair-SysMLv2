package CoreDataTypes {
    private import ScalarValues::*;

    enum def StatusColor {
        enum red;
        enum yellow;
        enum green;
    }

    enum def DiameterSize {
        enum small;
        enum medium;
        enum large;
    }

    part def UnsignedInteger {
        attribute value : Integer;
        constraint nonNegative {
            value >= 0
        }
    }

    part def UnsignedFloat {
        attribute value : Real;
        constraint nonNegative {
            value >= 0.0
        }
    }

    part def Status {
        attribute level : StatusColor;
        attribute color : String = switch (level) {
            case (StatusColor::red) "red";
            case (StatusColor::yellow) "yellow";
            case (StatusColor::green) "green";
        };
    }

    part def PhysicalDimension {
        attribute diameter : DiameterSize = DiameterSize::small;
    }

    part def SystemData {
        attribute unsignedInt : UnsignedInteger;
        attribute unsignedFloat : UnsignedFloat;
        attribute dateTime : DateTime;
        attribute text : String;
        attribute flag : Boolean;
        attribute status : Status;
        attribute dimension : PhysicalDimension;
    }

    part system : SystemData;
}