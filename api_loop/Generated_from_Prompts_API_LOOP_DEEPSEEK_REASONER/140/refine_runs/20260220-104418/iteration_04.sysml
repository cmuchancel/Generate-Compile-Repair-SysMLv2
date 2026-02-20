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
        part meter : Unit;
        part kilogram : Unit;
        part second : Unit;
        part ampere : Unit;
        part kelvin : Unit;
        part mole : Unit;
        part candela : Unit;
    }

    item def PhysicalQuantity {
        attribute magnitude : Real;
        part unit : Unit;
    }

    part def QuantityManager {
        part baseUnits : SIUnits;
        attribute quantities : PhysicalQuantity[0..*];
    }

    part meterInstance : Unit;
    part kilogramInstance : Unit;
    part secondInstance : Unit;
    part ampereInstance : Unit;
    part kelvinInstance : Unit;
    part moleInstance : Unit;
    part candelaInstance : Unit;

    part units : SIUnits;
    part manager : QuantityManager;

    connect units.meter to meterInstance;
    connect units.kilogram to kilogramInstance;
    connect units.second to secondInstance;
    connect units.ampere to ampereInstance;
    connect units.kelvin to kelvinInstance;
    connect units.mole to moleInstance;
    connect units.candela to candelaInstance;
    connect manager.baseUnits to units;
    
    attribute meterInstance.symbol : string = "m";
    attribute meterInstance.quantityKind : BaseQuantityKind = BaseQuantityKind::length;
    attribute meterInstance.conversionFactor : Real = 1.0;
    
    attribute kilogramInstance.symbol : string = "kg";
    attribute kilogramInstance.quantityKind : BaseQuantityKind = BaseQuantityKind::mass;
    attribute kilogramInstance.conversionFactor : Real = 1.0;
    
    attribute secondInstance.symbol : string = "s";
    attribute secondInstance.quantityKind : BaseQuantityKind = BaseQuantityKind::time;
    attribute secondInstance.conversionFactor : Real = 1.0;
    
    attribute ampereInstance.symbol : string = "A";
    attribute ampereInstance.quantityKind : BaseQuantityKind = BaseQuantityKind::electricCurrent;
    attribute ampereInstance.conversionFactor : Real = 1.0;
    
    attribute kelvinInstance.symbol : string = "K";
    attribute kelvinInstance.quantityKind : BaseQuantityKind = BaseQuantityKind::temperature;
    attribute kelvinInstance.conversionFactor : Real = 1.0;
    
    attribute moleInstance.symbol : string = "mol";
    attribute moleInstance.quantityKind : BaseQuantityKind = BaseQuantityKind::amountOfSubstance;
    attribute moleInstance.conversionFactor : Real = 1.0;
    
    attribute candelaInstance.symbol : string = "cd";
    attribute candelaInstance.quantityKind : BaseQuantityKind = BaseQuantityKind::luminousIntensity;
    attribute candelaInstance.conversionFactor : Real = 1.0;
}