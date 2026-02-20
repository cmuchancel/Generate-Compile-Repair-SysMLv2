package AutomotiveDomain {
    private import ScalarValues::*;

    /* Core automobile definition with alias */
    part def 汽车 {
        alias: 汽车 (Car);
        doc /* The central component representing an automobile in the model.
             This part definition serves as the foundation for all automotive-related elements. */

        /* Physical quantities associated with automobiles */
        attribute 扭矩 : TorqueValue;
        attribute 最高速度 : SpeedValue;
        attribute 功率 : PowerValue;
        attribute 油耗 : FuelConsumptionValue;
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
    part myCar : 汽车 {
        attribute 扭矩 : TorqueValue = (value => 350, unit => Nm);
        attribute 最高速度 : SpeedValue = (value => 250, unit => km_h);
        attribute 功率 : PowerValue = (value => 200, unit => kW);
        attribute 油耗 : FuelConsumptionValue = (value => 6.5, unit => L_100km);
    }
}