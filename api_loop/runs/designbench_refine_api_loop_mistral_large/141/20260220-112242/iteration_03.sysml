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
        attribute name : String;
        attribute symbol : String;
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
        attribute name = "meter";
        attribute symbol = "m";
        attribute quantityKind = length;
    }

    part kilogram : Unit {
        attribute name = "kilogram";
        attribute symbol = "kg";
        attribute quantityKind = mass;
    }

    part second : Unit {
        attribute name = "second";
        attribute symbol = "s";
        attribute quantityKind = time;
    }

    part ampere : Unit {
        attribute name = "ampere";
        attribute symbol = "A";
        attribute quantityKind = electricCurrent;
    }

    part kelvin : Unit {
        attribute name = "kelvin";
        attribute symbol = "K";
        attribute quantityKind = temperature;
    }

    part mole : Unit {
        attribute name = "mole";
        attribute symbol = "mol";
        attribute quantityKind = amountOfSubstance;
    }

    part candela : Unit {
        attribute name = "candela";
        attribute symbol = "cd";
        attribute quantityKind = luminousIntensity;
    }

    part inch : Unit {
        attribute name = "inch";
        attribute symbol = "in";
        attribute quantityKind = length;
    }

    part foot : Unit {
        attribute name = "foot";
        attribute symbol = "ft";
        attribute quantityKind = length;
    }

    part yard : Unit {
        attribute name = "yard";
        attribute symbol = "yd";
        attribute quantityKind = length;
    }

    part mile : Unit {
        attribute name = "mile";
        attribute symbol = "mi";
        attribute quantityKind = length;
    }

    part poundMass : Unit {
        attribute name = "pound";
        attribute symbol = "lbm";
        attribute quantityKind = mass;
    }

    part ounce : Unit {
        attribute name = "ounce";
        attribute symbol = "oz";
        attribute quantityKind = mass;
    }

    part rankine : Unit {
        attribute name = "rankine";
        attribute symbol = "°R";
        attribute quantityKind = temperature;
    }

    part fahrenheit : Unit {
        attribute name = "fahrenheit";
        attribute symbol = "°F";
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