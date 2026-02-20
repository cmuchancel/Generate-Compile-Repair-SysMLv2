package PhysicalQuantitySystem {
    private import ScalarValues::*;
    
    enum def BaseQuantity {
        enum length,
        enum mass,
        enum time,
        enum electricCurrent,
        enum thermodynamicTemperature,
        enum amountOfSubstance,
        enum luminousIntensity
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
        symbol = "m";
        quantity = BaseQuantity::length;
        conversionFactor = 1.0;
    }
    
    part kilometer : Unit {
        symbol = "km";
        quantity = BaseQuantity::length;
        conversionFactor = 1000.0;
    }
    
    part kilogram : Unit {
        symbol = "kg";
        quantity = BaseQuantity::mass;
        conversionFactor = 1.0;
    }
    
    part second : Unit {
        symbol = "s";
        quantity = BaseQuantity::time;
        conversionFactor = 1.0;
    }
    
    part ampere : Unit {
        symbol = "A";
        quantity = BaseQuantity::electricCurrent;
        conversionFactor = 1.0;
    }
    
    part kelvin : Unit {
        symbol = "K";
        quantity = BaseQuantity::thermodynamicTemperature;
        conversionFactor = 1.0;
    }
    
    part mole : Unit {
        symbol = "mol";
        quantity = BaseQuantity::amountOfSubstance;
        conversionFactor = 1.0;
    }
    
    part candela : Unit {
        symbol = "cd";
        quantity = BaseQuantity::luminousIntensity;
        conversionFactor = 1.0;
    }
    
    part manager : QuantityManager;
    
    port def QuantityPort;
    
    port managerPort : QuantityPort;
    
    connect manager to managerPort;
}