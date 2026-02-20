package AutomotiveDomain {
    private import ScalarValues::*;

    /* Core automobile definition with alias */
    part def 汽车 {
        attribute name : String = "汽车 (Car)";
        doc /* Core component representing an automobile with standardized characteristics. */
    }

    /* Physical quantities for automotive characteristics */
    part def 扭矩 {
        attribute value : Real;
        attribute unit : String = "N·m";
        doc /* Torque measurement for automotive systems. */
    }

    /* Supporting automotive elements */
    part def 发动机 {
        attribute type : String;
        doc /* Engine component of an automobile. */
    }

    part def 变速箱 {
        attribute gearRatio : Real;
        doc /* Transmission system of an automobile. */
    }

    /* Example automobile instance */
    part myCar : 汽车 {
        part engine : 发动机;
        part transmission : 变速箱;
        attribute torque : 扭矩 = (value => 350, unit => "N·m");
    }

    /* Standardized automotive interfaces */
    port def 动力输出 {
        doc /* Standard interface for power output. */
    }

    port def 动力输入 {
        doc /* Standard interface for power input. */
    }

    /* Example connections */
    connect myCar.engine to myCar.transmission using 动力输出;
}