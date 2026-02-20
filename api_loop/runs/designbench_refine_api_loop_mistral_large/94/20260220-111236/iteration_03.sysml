package AutomotiveDomain {
    private import ScalarValues::*;

    /* Core automobile definition with alias */
    part def Car {
        doc /* The central component representing an automobile in the model. */
        attribute alias : String = "汽车 (Car)";
    }

    /* Physical quantities for automotive characteristics */
    attribute def Torque : Real;
    doc /* Torque, measured in Newton-meters (Nm). */ for Torque;

    attribute def Speed : Real;
    doc /* Speed, measured in kilometers per hour (km/h). */ for Speed;

    attribute def Power : Real;
    doc /* Power, measured in kilowatts (kW). */ for Power;

    attribute def FuelConsumption : Real;
    doc /* Fuel consumption, measured in liters per 100 kilometers (L/100km). */ for FuelConsumption;

    /* Example automobile instance */
    part myCar : Car {
        attribute torque : Real = 350.0;
        attribute speed : Real = 0.0;
        attribute power : Real = 150.0;
        attribute fuelConsumption : Real = 6.5;
    }
}