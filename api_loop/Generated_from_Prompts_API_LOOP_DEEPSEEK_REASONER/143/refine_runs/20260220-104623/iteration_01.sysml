package SystemModel {
    private import ScalarValues::*;

    item def UnsignedInteger {
        attribute value : Integer;
    }

    item def UnsignedFloat {
        attribute value : Real;
    }

    enum def StatusCondition {
        critical : "red";
        warning : "yellow";
        normal : "green";
    }

    enum def DiameterOption {
        Small : 60;
        Medium : 70;
        Large : 80;
    }

    part def DataManager {
        attribute id : UnsignedInteger;
        attribute measurement : UnsignedFloat;
        attribute timestamp : DateTime;
        attribute label : String;
        attribute active : Boolean;
        attribute status : StatusCondition;
        attribute diameter : DiameterOption = DiameterOption::Small;
    }
}