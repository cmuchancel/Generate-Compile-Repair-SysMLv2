package AutomotiveDomain {
    private import ScalarValues::*;

    /* Core automobile definition with alias */
    part def Automobile {
        alias Car;
        comment /* The central component representing an automobile in the model.
             This part definition serves as the foundation for all automotive-related elements. */

        /* Physical quantities associated with automobiles */
        attribute torque : TorqueValue;
        attribute maxSpeed : SpeedValue;
        attribute power : PowerValue;
        attribute fuelConsumption : FuelConsumptionValue;
    }

    /* Physical quantity definitions */
    part def TorqueValue {
        comment /* Standardized representation of torque in automotive systems */
        attribute value : Real;
        attribute unit : TorqueUnit;
    }

    part def SpeedValue {
        comment /* Standardized representation of speed */
        attribute value : Real;
        attribute unit : SpeedUnit;
    }

    part def PowerValue {
        comment /* Standardized representation of power */
        attribute value : Real;
        attribute unit : PowerUnit;
    }

    part def FuelConsumptionValue {
        comment /* Standardized representation of fuel consumption */
        attribute value : Real;
        attribute unit : FuelConsumptionUnit;
    }

    /* Unit enumerations */
    enum def TorqueUnit {
        enum Nm;
        enum lb_ft;
    }

    enum def SpeedUnit {
        enum km_h;
        enum mph;
    }

    enum def PowerUnit {
        enum kW;
        enum hp;
    }

    enum def FuelConsumptionUnit {
        enum L_100km;
        enum mpg;
    }

    /* Example automobile instance */
    part myCar :> Automobile {
        attribute torque : TorqueValue {
            attribute torqueValue : Real = 350;
            attribute torqueUnit : TorqueUnit = Nm;
        }
        attribute maxSpeed : SpeedValue {
            attribute speedValue : Real = 250;
            attribute speedUnit : SpeedUnit = km_h;
        }
        attribute power : PowerValue {
            attribute powerValue : Real = 200;
            attribute powerUnit : PowerUnit = kW;
        }
        attribute fuelConsumption : FuelConsumptionValue {
            attribute fuelValue : Real = 6.5;
            attribute fuelUnit : FuelConsumptionUnit = L_100km;
        }
    }
}