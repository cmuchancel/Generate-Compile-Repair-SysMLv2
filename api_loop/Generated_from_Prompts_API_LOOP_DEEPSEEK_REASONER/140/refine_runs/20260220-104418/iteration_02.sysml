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

    part meter : Unit {
        attribute symbol = "m";
        attribute quantityKind = BaseQuantityKind::length;
        attribute conversionFactor = 1.0;
    }

    part kilogram : Unit {
        attribute symbol = "kg";
        attribute quantityKind = BaseQuantityKind::mass;
        attribute conversionFactor = 1.0;
    }

    part second : Unit {
        attribute symbol = "s";
        attribute quantityKind = BaseQuantityKind::time;
        attribute conversionFactor = 1.0;
    }

    part ampere : Unit {
        attribute symbol = "A";
        attribute quantityKind = BaseQuantityKind::electricCurrent;
        attribute conversionFactor = 1.0;
    }

    part kelvin : Unit {
        attribute symbol = "K";
        attribute quantityKind = BaseQuantityKind::temperature;
        attribute conversionFactor = 1.0;
    }

    part mole : Unit {
        attribute symbol = "mol";
        attribute quantityKind = BaseQuantityKind::amountOfSubstance;
        attribute conversionFactor = 1.0;
    }

    part candela : Unit {
        attribute symbol = "cd";
        attribute quantityKind = BaseQuantityKind::luminousIntensity;
        attribute conversionFactor = 1.0;
    }

    part units : SIUnits;
    part manager : QuantityManager;

    connect units.meter to meter;
    connect units.kilogram to kilogram;
    connect units.second to second;
    connect units.ampere to ampere;
    connect units.kelvin to kelvin;
    connect units.mole to mole;
    connect units.candela to candela;
    connect manager.baseUnits to units;
}