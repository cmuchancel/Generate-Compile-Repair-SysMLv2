package AutomobileManagement;

private import ScalarValues::*;

item def TorqueValue : Real;
item def Torque : TorqueValue;

part def Automobile {
    attribute torque : Torque;
}

part def Car : Automobile;