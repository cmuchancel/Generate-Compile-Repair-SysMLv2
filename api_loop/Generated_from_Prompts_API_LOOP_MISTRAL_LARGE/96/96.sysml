package AutomobileModel {
    private import ScalarValues::*;

    part def Automobile;

    part def TorqueValue {
        attribute value : Real;
        attribute unit : String = "N·m";
    }

    part def AutomobileSystem {
        part car : Automobile;
        attribute engineTorque : TorqueValue;
    }
}