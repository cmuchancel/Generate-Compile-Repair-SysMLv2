package AutomotiveDomain {
    private import ScalarValues::*;

    /* Core automobile definition with alias */
    part def 汽车 {
        alias: 汽车 (Car);
        doc /* The central component representing an automobile in the model. */
    }

    /* Physical quantities for automotive characteristics */
    attribute def 扭矩 : ScalarValues::Real {
        doc /* Torque, measured in Newton-meters (Nm). */
    }

    /* Additional automotive physical quantities */
    attribute def 速度 : ScalarValues::Real {
        doc /* Speed, measured in kilometers per hour (km/h). */
    }

    attribute def 功率 : ScalarValues::Real {
        doc /* Power, measured in kilowatts (kW). */
    }

    attribute def 油耗 : ScalarValues::Real {
        doc /* Fuel consumption, measured in liters per 100 kilometers (L/100km). */
    }

    /* Example automobile instance */
    part myCar : 汽车 {
        attribute torque : 扭矩 = 350.0;
        attribute speed : 速度 = 0.0;
        attribute power : 功率 = 150.0;
        attribute fuelConsumption : 油耗 = 6.5;
    }
}