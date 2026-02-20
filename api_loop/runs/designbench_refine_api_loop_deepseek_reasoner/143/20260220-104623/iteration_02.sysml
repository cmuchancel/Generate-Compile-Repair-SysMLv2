package SystemModel {
    private import ScalarValues::*;

    item def UnsignedInteger {
        attribute value : Integer;
    }

    item def UnsignedFloat {
        attribute value : Real;
    }

    enum def StatusCondition {
        critical;
        warning;
        normal;
    }

    part def DataManager {
        attribute id : UnsignedInteger;
        attribute measurement : UnsignedFloat;
        attribute timestamp : String;
        attribute label : String;
        attribute active : Boolean;
        attribute status : StatusCondition;
        attribute diameter : Integer = 60;
    }
}