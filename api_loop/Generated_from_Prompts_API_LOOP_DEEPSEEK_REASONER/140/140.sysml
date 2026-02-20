package PhysicalQuantitySystem {
    private import ScalarValues::*;

    enum def BaseQuantity {
        enum length;
        enum mass;
        enum time;
        enum electricCurrent;
        enum thermodynamicTemperature;
        enum amountOfSubstance;
        enum luminousIntensity;
    }

    part def Unit {
        attribute symbol : String;
        attribute quantity : BaseQuantity;
        attribute conversionFactor : Real = 1.0;
    }

    part def PhysicalQuantity {
        attribute value : Real;
        attribute unit : Unit;
    }

    part def QuantityManager {
        attribute quantities : PhysicalQuantity[0..*];
        attribute availableUnits : Unit[0..*];
    }

    part meter : Unit {
        attribute symbol = "m";
        attribute quantity = BaseQuantity::length;
        attribute conversionFactor = 1.0;
    }

    part kilometer : Unit {
        attribute symbol = "km";
        attribute quantity = BaseQuantity::length;
        attribute conversionFactor = 1000.0;
    }

    part kilogram : Unit {
        attribute symbol = "kg";
        attribute quantity = BaseQuantity::mass;
        attribute conversionFactor = 1.0;
    }

    part second : Unit {
        attribute symbol = "s";
        attribute quantity = BaseQuantity::time;
        attribute conversionFactor = 1.0;
    }

    part ampere : Unit {
        attribute symbol = "A";
        attribute quantity = BaseQuantity::electricCurrent;
        attribute conversionFactor = 1.0;
    }

    part kelvin : Unit {
        attribute symbol = "K";
        attribute quantity = BaseQuantity::thermodynamicTemperature;
        attribute conversionFactor = 1.0;
    }

    part mole : Unit {
        attribute symbol = "mol";
        attribute quantity = BaseQuantity::amountOfSubstance;
        attribute conversionFactor = 1.0;
    }

    part candela : Unit {
        attribute symbol = "cd";
        attribute quantity = BaseQuantity::luminousIntensity;
        attribute conversionFactor = 1.0;
    }

    part manager : QuantityManager;

    port def QuantityPort;

    port managerPort : QuantityPort;

    connect manager to managerPort;
}