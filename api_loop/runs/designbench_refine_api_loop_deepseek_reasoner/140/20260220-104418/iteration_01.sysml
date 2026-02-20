package PhysicalQuantitySystem {
    private import ScalarValues::*;
    
    enum def BaseQuantityKind {
        enum length;
        enum mass;
        enum time;
        enum electricCurrent;
        enum temperature;
        enum amountOfSubstance;
        enum luminousIntensity;
    }
    
    item def Unit {
        attribute symbol : string;
        attribute quantityKind : BaseQuantityKind;
        attribute conversionFactor : Real;
    }
    
    part def SIUnits {
        attribute meter : Unit = Unit{symbol = "m", quantityKind = BaseQuantityKind::length, conversionFactor = 1.0};
        attribute kilogram : Unit = Unit{symbol = "kg", quantityKind = BaseQuantityKind::mass, conversionFactor = 1.0};
        attribute second : Unit = Unit{symbol = "s", quantityKind = BaseQuantityKind::time, conversionFactor = 1.0};
        attribute ampere : Unit = Unit{symbol = "A", quantityKind = BaseQuantityKind::electricCurrent, conversionFactor = 1.0};
        attribute kelvin : Unit = Unit{symbol = "K", quantityKind = BaseQuantityKind::temperature, conversionFactor = 1.0};
        attribute mole : Unit = Unit{symbol = "mol", quantityKind = BaseQuantityKind::amountOfSubstance, conversionFactor = 1.0};
        attribute candela : Unit = Unit{symbol = "cd", quantityKind = BaseQuantityKind::luminousIntensity, conversionFactor = 1.0};
    }
    
    item def PhysicalQuantity {
        attribute magnitude : Real;
        attribute unit : Unit;
    }
    
    part def QuantityManager {
        attribute baseUnits : SIUnits;
        attribute quantities : PhysicalQuantity[0..*];
    }
    
    part units : SIUnits;
    part manager : QuantityManager;
    
    connect manager.baseUnits to units;
}