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

    part meterInstance : Unit {
        attribute symbol = "m";
        attribute quantityKind = BaseQuantityKind::length;
        attribute conversionFactor = 1.0;
    }

    part kilogramInstance : Unit {
        attribute symbol = "kg";
        attribute quantityKind = BaseQuantityKind::mass;
        attribute conversionFactor = 1.0;
    }

    part secondInstance : Unit {
        attribute symbol = "s";
        attribute quantityKind = BaseQuantityKind::time;
        attribute conversionFactor = 1.0;
    }

    part ampereInstance : Unit {
        attribute symbol = "A";
        attribute quantityKind = BaseQuantityKind::electricCurrent;
        attribute conversionFactor = 1.0;
    }

    part kelvinInstance : Unit {
        attribute symbol = "K";
        attribute quantityKind = BaseQuantityKind::temperature;
        attribute conversionFactor = 1.0;
    }

    part moleInstance : Unit {
        attribute symbol = "mol";
        attribute quantityKind = BaseQuantityKind::amountOfSubstance;
        attribute conversionFactor = 1.0;
    }

    part candelaInstance : Unit {
        attribute symbol = "cd";
        attribute quantityKind = BaseQuantityKind::luminousIntensity;
        attribute conversionFactor = 1.0;
    }

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
}