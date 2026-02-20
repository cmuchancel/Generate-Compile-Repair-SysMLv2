package AutomobileSystem {
    private import ScalarValues::*;

    part def Automobile;
    part def Car :> Automobile;

    attribute def TorqueValue : Real;
    attribute def Torque :> TorqueValue;

    part myCar : Car {
        attribute torque : Torque;
    }
}