package AutomotiveDomain {
    private import ScalarValues::*;

    /* Core automobile definition with alias */
    part def Automobile {
        attribute name : String = "汽车 (Car)";
        doc /* Core component representing an automobile with standardized characteristics. */
    }

    /* Physical quantities for automotive characteristics */
    part def Torque {
        attribute value : Real;
        attribute unit : String = "N·m";
        doc /* Torque measurement for automotive systems. */
    }

    /* Supporting automotive elements */
    part def Engine {
        attribute type : String;
        doc /* Engine component of an automobile. */
    }

    part def Transmission {
        attribute gearRatio : Real;
        doc /* Transmission system of an automobile. */
    }

    /* Example automobile instance */
    part myCar {
        part :>> Automobile;
        part engine :>> Engine;
        part transmission :>> Transmission;
        attribute torque : Torque = (value: 350, unit: "N·m");
    }

    /* Standardized automotive interfaces */
    port def PowerOutput {
        doc /* Standard interface for power output. */
    }

    port def PowerInput {
        doc /* Standard interface for power input. */
    }

    /* Example connections */
    connect myCar.engine to myCar.transmission using PowerOutput;
}