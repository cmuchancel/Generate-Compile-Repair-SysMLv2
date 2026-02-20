package UnitSystemLibrary {
    private import ScalarValues::*;

    enum def QuantityKind {
        enum length;
        enum mass;
        enum time;
        enum electricCurrent;
        enum thermodynamicTemperature;
        enum amountOfSubstance;
        enum luminousIntensity;
    }

    enum def ISQUnit {
        enum meter :> length;
        enum kilogram :> mass;
        enum second :> time;
        enum ampere :> electricCurrent;
        enum kelvin :> thermodynamicTemperature;
        enum mole :> amountOfSubstance;
        enum candela :> luminousIntensity;
    }

    enum def USCustomaryUnit {
        enum inch :> length;
        enum foot :> length;
        enum yard :> length;
        enum mile :> length;
        enum poundMass :> mass;
        enum ounce :> mass;
        enum second :> time;
        enum ampere :> electricCurrent;
        enum rankine :> thermodynamicTemperature;
        enum fahrenheit :> thermodynamicTemperature;
    }

    part def Unit {
    }

    part def UnitConverter {
        attribute sourceUnit : Unit;
        attribute targetUnit : Unit;
        attribute conversionFactor : Real;
        attribute offset : Real = 0.0;
    }

    part def UnitSystem {
        attribute name : String;
        attribute baseUnits : Unit[0..*];
    }

    part def PhysicalQuantity {
        attribute value : Real;
        attribute unit : Unit;
        attribute quantityKind : QuantityKind;
    }

    part meter : Unit {
    }

    part kilogram : Unit {
    }

    part second : Unit {
    }

    part ampere : Unit {
    }

    part kelvin : Unit {
    }

    part mole : Unit {
    }

    part candela : Unit {
    }

    part inch : Unit {
    }

    part foot : Unit {
    }

    part yard : Unit {
    }

    part mile : Unit {
    }

    part poundMass : Unit {
    }

    part ounce : Unit {
    }

    part rankine : Unit {
    }

    part fahrenheit : Unit {
    }

    part isqSystem : UnitSystem {
        attribute name = "ISQ";
        attribute baseUnits = (meter, kilogram, second, ampere, kelvin, mole, candela);
    }

    part usCustomarySystem : UnitSystem {
        attribute name = "USCustomary";
        attribute baseUnits = (foot, poundMass, second, ampere, rankine);
    }

    part lengthConverter : UnitConverter {
        attribute sourceUnit = meter;
        attribute targetUnit = foot;
        attribute conversionFactor = 3.28084;
    }

    part massConverter : UnitConverter {
        attribute sourceUnit = kilogram;
        attribute targetUnit = poundMass;
        attribute conversionFactor = 2.20462;
    }

    part temperatureConverter : UnitConverter {
        attribute sourceUnit = kelvin;
        attribute targetUnit = rankine;
        attribute conversionFactor = 1.8;
        attribute offset = 0.0;
    }

    part fahrenheitConverter : UnitConverter {
        attribute sourceUnit = kelvin;
        attribute targetUnit = fahrenheit;
        attribute conversionFactor = 1.8;
        attribute offset = 459.67;
    }
}