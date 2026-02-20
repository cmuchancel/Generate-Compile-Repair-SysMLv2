package AutomobileModel {
    private import ScalarValues::*;

    part def Automobile;
    alias def Car = Automobile;

    item def TorqueValue {
        attribute value : Real;
        attribute unit : String = "N·m";
    }
    alias def Torque = TorqueValue;

    part def AutomobileSystem {
        part car : Automobile;
        attribute engineTorque : Torque;
    }
}