package UnitSystemLibrary {
    private import ScalarValues::*;

    enum def QuantityKind {
        enum length;
        enum mass;
        enum time;
        enum electricCurrent;
        enum temperature;
        enum amountOfSubstance;
        enum luminousIntensity;
    }

    enum def ISQUnit {
        enum meter :> length;
        enum kilogram :> mass;
        enum second :> time;
        enum ampere :> electricCurrent;
        enum kelvin :> temperature;
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
        enum rankine :> temperature;
        enum fahrenheit :> temperature;
    }

    part def Unit {
        attribute unitName : String;
        attribute unitSymbol : String;
        attribute quantityKind : QuantityKind;
    }

    part def UnitConverter {
        attribute sourceUnit : Unit;
        attribute targetUnit : Unit;
        attribute conversionFactor : Real;
        attribute offset : Real = 0.0;
    }

    part def UnitSystem {
        attribute systemName : String;
        attribute baseUnitNames : String[0..*];
    }

    part def PhysicalQuantity {
        attribute value : Real;
        attribute unit : Unit;
    }

    part meter : Unit {
        attribute unitName = "meter";
        attribute unitSymbol = "m";
        attribute quantityKind = length;
    }

    part kilogram : Unit {
        attribute unitName = "kilogram";
        attribute unitSymbol = "kg";
        attribute quantityKind = mass;
    }

    part second : Unit {
        attribute unitName = "second";
        attribute unitSymbol = "s";
        attribute quantityKind = time;
    }

    part ampere : Unit {
        attribute unitName = "ampere";
        attribute unitSymbol = "A";
        attribute quantityKind = electricCurrent;
    }

    part kelvin : Unit {
        attribute unitName = "kelvin";
        attribute unitSymbol = "K";
        attribute quantityKind = temperature;
    }

    part mole : Unit {
        attribute unitName = "mole";
        attribute unitSymbol = "mol";
        attribute quantityKind = amountOfSubstance;
    }

    part candela : Unit {
        attribute unitName = "candela";
        attribute unitSymbol = "cd";
        attribute quantityKind = luminousIntensity;
    }

    part inch : Unit {
        attribute unitName = "inch";
        attribute unitSymbol = "in";
        attribute quantityKind = length;
    }

    part foot : Unit {
        attribute unitName = "foot";
        attribute unitSymbol = "ft";
        attribute quantityKind = length;
    }

    part yard : Unit {
        attribute unitName = "yard";
        attribute unitSymbol = "yd";
        attribute quantityKind = length;
    }

    part mile : Unit {
        attribute unitName = "mile";
        attribute unitSymbol = "mi";
        attribute quantityKind = length;
    }

    part poundMass : Unit {
        attribute unitName = "pound";
        attribute unitSymbol = "lbm";
        attribute quantityKind = mass;
    }

    part ounce : Unit {
        attribute unitName = "ounce";
        attribute unitSymbol = "oz";
        attribute quantityKind = mass;
    }

    part rankine : Unit {
        attribute unitName = "rankine";
        attribute unitSymbol = "°R";
        attribute quantityKind = temperature;
    }

    part fahrenheit : Unit {
        attribute unitName = "fahrenheit";
        attribute unitSymbol = "°F";
        attribute quantityKind = temperature;
    }

    part isqSystem : UnitSystem {
        attribute systemName = "ISQ";
        attribute baseUnitNames = ("meter", "kilogram", "second", "ampere", "kelvin", "mole", "candela");
    }

    part usCustomarySystem : UnitSystem {
        attribute systemName = "USCustomary";
        attribute baseUnitNames = ("foot", "pound", "second", "ampere", "rankine");
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