package AutomobileModel {
    private import ScalarValues::*;

    part def Automobile {
        attribute torque : TorqueValue;
    }

    part def Car : Automobile;

    item def Torque : TorqueValue;

    part def AutomobileSystem {
        part vehicle : Automobile;
        part car : Car;
        attribute maxTorque : Torque;
    }
}