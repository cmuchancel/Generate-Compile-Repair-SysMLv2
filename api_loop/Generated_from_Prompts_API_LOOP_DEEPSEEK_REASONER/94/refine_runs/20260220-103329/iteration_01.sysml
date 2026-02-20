package AutomotiveDomain {
    private import ScalarValues::*;

    alias "汽车 (Car)" for 汽车;

    item def 扭矩 {
        attribute value : Real;
        attribute unit : String = "Nm";
    }

    part def 汽车 {
        attribute 扭矩 : 扭矩;
    }
}