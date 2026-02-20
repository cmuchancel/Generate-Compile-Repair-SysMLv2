package AutomobileManagement;

private import ScalarValues::*;

part def TorqueValue {
    attribute value : Real;
}

part def Torque :> TorqueValue;

part def Automobile {
    attribute torque : Torque;
}

part def Car :> Automobile;