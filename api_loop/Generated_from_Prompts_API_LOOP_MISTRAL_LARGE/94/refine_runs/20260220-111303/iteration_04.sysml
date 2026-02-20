package AutomotiveDomain {
    private import ScalarValues::*;

    /* Core automobile definition with alias */
    part def Automobile {
        alias Car;
        doc /* The central component representing an automobile in the model.
             This part definition serves as the foundation for all automotive-related elements. */

        /* Physical quantities associated with automobiles */
        attribute torque : TorqueValue;
        attribute maxSpeed : SpeedValue;
        attribute power : PowerValue;
        attribute fuelConsumption : FuelConsumptionValue;
    }

    /* Physical quantity definitions */
    part def TorqueValue {
        doc /* Standardized representation of torque in automotive systems */
        attribute value : Real;
        attribute unit : TorqueUnit;
    }

    part def SpeedValue {
        doc /* Standardized representation of speed */
        attribute value : Real;
        attribute unit : SpeedUnit;
    }

    part def PowerValue {
        doc /* Standardized representation of power */
        attribute value : Real;
        attribute unit : PowerUnit;
    }

    part def FuelConsumptionValue {
        doc /* Standardized representation of fuel consumption */
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
        attribute torqueValue : TorqueValue {
            attribute value : Real = 350;
            attribute unit : TorqueUnit = Nm;
        }
        attribute maxSpeedValue : SpeedValue {
            attribute value : Real = 250;
            attribute unit : SpeedUnit = km_h;
        }
        attribute powerValue : PowerValue {
            attribute value : Real = 200;
            attribute unit : PowerUnit = kW;
        }
        attribute fuelConsumptionValue : FuelConsumptionValue {
            attribute value : Real = 6.5;
            attribute unit : FuelConsumptionUnit = L_100km;
        }
    }
}