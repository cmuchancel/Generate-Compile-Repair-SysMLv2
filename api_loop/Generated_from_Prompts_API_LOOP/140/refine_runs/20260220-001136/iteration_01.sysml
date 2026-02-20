package PhysicalQuantityLibrary {
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

    part def Unit {
        attribute name : String;
        attribute symbol : String;
        attribute quantityKind : BaseQuantityKind;
        attribute conversionFactorToSI : Real;
    }

    part def Quantity {
        attribute value : Real;
        part unit : Unit;
    }

    part def UnitExtension {
        part customUnit : Unit;
        attribute description : String;
    }

    part def Meter specializes Unit {
        attribute name : String = "meter";
        attribute symbol : String = "m";
        attribute quantityKind : BaseQuantityKind = length;
        attribute conversionFactorToSI : Real = 1.0;
    }

    part def Kilogram specializes Unit {
        attribute name : String = "kilogram";
        attribute symbol : String = "kg";
        attribute quantityKind : BaseQuantityKind = mass;
        attribute conversionFactorToSI : Real = 1.0;
    }

    part def Second specializes Unit {
        attribute name : String = "second";
        attribute symbol : String = "s";
        attribute quantityKind : BaseQuantityKind = time;
        attribute conversionFactorToSI : Real = 1.0;
    }

    part def Ampere specializes Unit {
        attribute name : String = "ampere";
        attribute symbol : String = "A";
        attribute quantityKind : BaseQuantityKind = electricCurrent;
        attribute conversionFactorToSI : Real = 1.0;
    }

    part def Kelvin specializes Unit {
        attribute name : String = "kelvin";
        attribute symbol : String = "K";
        attribute quantityKind : BaseQuantityKind = temperature;
        attribute conversionFactorToSI : Real = 1.0;
    }

    part def Mole specializes Unit {
        attribute name : String = "mole";
        attribute symbol : String = "mol";
        attribute quantityKind : BaseQuantityKind = amountOfSubstance;
        attribute conversionFactorToSI : Real = 1.0;
    }

    part def Candela specializes Unit {
        attribute name : String = "candela";
        attribute symbol : String = "cd";
        attribute quantityKind : BaseQuantityKind = luminousIntensity;
        attribute conversionFactorToSI : Real = 1.0;
    }

    part def PhysicalQuantitySystem {
        part lengthUnit : Unit;
        part massUnit : Unit;
        part timeUnit : Unit;
        part electricCurrentUnit : Unit;
        part temperatureUnit : Unit;
        part amountOfSubstanceUnit : Unit;
        part luminousIntensityUnit : Unit;
        part quantityData : Quantity;
        part extension : UnitExtension;
    }

    part def SIQuantitySystem specializes PhysicalQuantitySystem {
        part lengthUnit : Meter;
        part massUnit : Kilogram;
        part timeUnit : Second;
        part electricCurrentUnit : Ampere;
        part temperatureUnit : Kelvin;
        part amountOfSubstanceUnit : Mole;
        part luminousIntensityUnit : Candela;
        part quantityData : Quantity;
        part extension : UnitExtension;
    }
}