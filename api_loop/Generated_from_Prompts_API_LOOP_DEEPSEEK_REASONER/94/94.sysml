package AutomotiveDomain {
    private import ScalarValues::*;

    // 扭矩 (Torque) definition
    item def Torque {
        attribute value : Real;
        attribute unit : String = "Nm";
    }

    // 汽车 (Automobile) definition
    part def Automobile {
        // 扭矩 (torque) attribute
        attribute torque : Torque;
    }

    // Aliases for communication and usage in various application scenarios
    alias Car for Automobile;
}