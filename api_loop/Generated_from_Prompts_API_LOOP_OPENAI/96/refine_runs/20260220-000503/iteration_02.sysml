package AutomobileModel {
    private import ScalarValues::*;

    item def TorqueValue;
    item def Torque specializes TorqueValue;

    part def Automobile {
        attribute torque : TorqueValue;
    }

    part def Car specializes Automobile { }

    part def AutomobileSystem {
        part vehicle : Automobile;
        part car : Car;
        attribute maxTorque : Torque;
    }
}