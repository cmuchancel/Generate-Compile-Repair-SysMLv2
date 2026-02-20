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

    part meterInstance : Unit {
        attribute unitName = "meter";
        attribute unitSymbol = "m";
        attribute quantityKind = QuantityKind::length;
    }

    part kilogramInstance : Unit {
        attribute unitName = "kilogram";
        attribute unitSymbol = "kg";
        attribute quantityKind = QuantityKind::mass;
    }

    part secondInstance : Unit {
        attribute unitName = "second";
        attribute unitSymbol = "s";
        attribute quantityKind = QuantityKind::time;
    }

    part ampereInstance : Unit {
        attribute unitName = "ampere";
        attribute unitSymbol = "A";
        attribute quantityKind = QuantityKind::electricCurrent;
    }

    part kelvinInstance : Unit {
        attribute unitName = "kelvin";
        attribute unitSymbol = "K";
        attribute quantityKind = QuantityKind::temperature;
    }

    part moleInstance : Unit {
        attribute unitName = "mole";
        attribute unitSymbol = "mol";
        attribute quantityKind = QuantityKind::amountOfSubstance;
    }

    part candelaInstance : Unit {
        attribute unitName = "candela";
        attribute unitSymbol = "cd";
        attribute quantityKind = QuantityKind::luminousIntensity;
    }

    part inchInstance : Unit {
        attribute unitName = "inch";
        attribute unitSymbol = "in";
        attribute quantityKind = QuantityKind::length;
    }

    part footInstance : Unit {
        attribute unitName = "foot";
        attribute unitSymbol = "ft";
        attribute quantityKind = QuantityKind::length;
    }

    part yardInstance : Unit {
        attribute unitName = "yard";
        attribute unitSymbol = "yd";
        attribute quantityKind = QuantityKind::length;
    }

    part mileInstance : Unit {
        attribute unitName = "mile";
        attribute unitSymbol = "mi";
        attribute quantityKind = QuantityKind::length;
    }

    part poundMassInstance : Unit {
        attribute unitName = "pound";
        attribute unitSymbol = "lbm";
        attribute quantityKind = QuantityKind::mass;
    }

    part ounceInstance : Unit {
        attribute unitName = "ounce";
        attribute unitSymbol = "oz";
        attribute quantityKind = QuantityKind::mass;
    }

    part rankineInstance : Unit {
        attribute unitName = "rankine";
        attribute unitSymbol = "°R";
        attribute quantityKind = QuantityKind::temperature;
    }

    part fahrenheitInstance : Unit {
        attribute unitName = "fahrenheit";
        attribute unitSymbol = "°F";
        attribute quantityKind = QuantityKind::temperature;
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
        attribute sourceUnit = meterInstance;
        attribute targetUnit = footInstance;
        attribute conversionFactor = 3.28084;
    }

    part massConverter : UnitConverter {
        attribute sourceUnit = kilogramInstance;
        attribute targetUnit = poundMassInstance;
        attribute conversionFactor = 2.20462;
    }

    part temperatureConverter : UnitConverter {
        attribute sourceUnit = kelvinInstance;
        attribute targetUnit = rankineInstance;
        attribute conversionFactor = 1.8;
        attribute offset = 0.0;
    }

    part fahrenheitConverter : UnitConverter {
        attribute sourceUnit = kelvinInstance;
        attribute targetUnit = fahrenheitInstance;
        attribute conversionFactor = 1.8;
        attribute offset = 459.67;
    }
}