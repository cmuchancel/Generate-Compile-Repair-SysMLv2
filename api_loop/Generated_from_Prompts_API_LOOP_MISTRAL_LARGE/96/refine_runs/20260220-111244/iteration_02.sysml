package AutomobileModel {
    private import ScalarValues::*;

    part def Automobile;
    alias Car = Automobile;

    item def TorqueValue {
        attribute value : Real;
        attribute unit : String = "N·m";
    }
    alias Torque = TorqueValue;

    part def AutomobileSystem {
        part car : Automobile;
        attribute engineTorque : Torque;
    }
}