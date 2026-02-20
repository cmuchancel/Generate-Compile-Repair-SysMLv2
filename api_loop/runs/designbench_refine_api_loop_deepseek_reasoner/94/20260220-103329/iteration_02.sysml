package AutomotiveDomain {
    private import BaseTypes::*;

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
    alias 扭矩 for Torque;
    alias 汽车 for Automobile;
    alias Car for Automobile;
}