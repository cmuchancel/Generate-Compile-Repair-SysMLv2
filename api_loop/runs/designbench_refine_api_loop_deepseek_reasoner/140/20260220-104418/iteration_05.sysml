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
        part meter : Unit {
            attribute symbol : string = "m";
            attribute quantityKind : BaseQuantityKind = BaseQuantityKind::length;
            attribute conversionFactor : Real = 1.0;
        }
        part kilogram : Unit {
            attribute symbol : string = "kg";
            attribute quantityKind : BaseQuantityKind = BaseQuantityKind::mass;
            attribute conversionFactor : Real = 1.0;
        }
        part second : Unit {
            attribute symbol : string = "s";
            attribute quantityKind : BaseQuantityKind = BaseQuantityKind::time;
            attribute conversionFactor : Real = 1.0;
        }
        part ampere : Unit {
            attribute symbol : string = "A";
            attribute quantityKind : BaseQuantityKind = BaseQuantityKind::electricCurrent;
            attribute conversionFactor : Real = 1.0;
        }
        part kelvin : Unit {
            attribute symbol : string = "K";
            attribute quantityKind : BaseQuantityKind = BaseQuantityKind::temperature;
            attribute conversionFactor : Real = 1.0;
        }
        part mole : Unit {
            attribute symbol : string = "mol";
            attribute quantityKind : BaseQuantityKind = BaseQuantityKind::amountOfSubstance;
            attribute conversionFactor : Real = 1.0;
        }
        part candela : Unit {
            attribute symbol : string = "cd";
            attribute quantityKind : BaseQuantityKind = BaseQuantityKind::luminousIntensity;
            attribute conversionFactor : Real = 1.0;
        }
    }

    item def PhysicalQuantity {
        attribute magnitude : Real;
        part unit : Unit;
    }

    part def QuantityManager {
        part baseUnits : SIUnits;
        attribute quantities : PhysicalQuantity[0..*];
    }

    part units : SIUnits;
    part manager : QuantityManager;

    connect manager.baseUnits to units;
}