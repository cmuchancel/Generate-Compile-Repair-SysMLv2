package AutomotiveDomain {
    private import ScalarValues::*;

    /* Core automobile definition with alias */
    part def Car {
        attribute alias : String = "汽车 (Car)";
        doc /* The central component representing an automobile in the model. */
    }

    /* Physical quantities for automotive characteristics */
    attribute def Torque : Real {
        doc /* Torque, measured in Newton-meters (Nm). */
    }

    /* Additional automotive physical quantities */
    attribute def Speed : Real {
        doc /* Speed, measured in kilometers per hour (km/h). */
    }

    attribute def Power : Real {
        doc /* Power, measured in kilowatts (kW). */
    }

    attribute def FuelConsumption : Real {
        doc /* Fuel consumption, measured in liters per 100 kilometers (L/100km). */
    }

    /* Example automobile instance */
    part myCar : Car {
        attribute torque : Torque = 350.0;
        attribute speed : Speed = 0.0;
        attribute power : Power = 150.0;
        attribute fuelConsumption : FuelConsumption = 6.5;
    }
}