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

    part def UnitConverter {
        attribute sourceUnit : ISQUnit | USCustomaryUnit;
        attribute targetUnit : ISQUnit | USCustomaryUnit;
        attribute conversionFactor : Real;
        attribute offset : Real = 0.0;
    }

    part def UnitSystem {
        attribute name : String;
        attribute baseUnits : ISQUnit[0..*] | USCustomaryUnit[0..*];
    }

    part def PhysicalQuantity {
        attribute value : Real;
        attribute unit : ISQUnit | USCustomaryUnit;
        attribute quantityKind : QuantityKind;
    }

    part isqSystem : UnitSystem {
        attribute name = "ISQ";
        attribute baseUnits = ISQUnit::meter, ISQUnit::kilogram, ISQUnit::second, ISQUnit::ampere, ISQUnit::kelvin, ISQUnit::mole, ISQUnit::candela;
    }

    part usCustomarySystem : UnitSystem {
        attribute name = "USCustomary";
        attribute baseUnits = USCustomaryUnit::foot, USCustomaryUnit::poundMass, USCustomaryUnit::second, USCustomaryUnit::ampere, USCustomaryUnit::rankine;
    }

    part lengthConverter : UnitConverter {
        attribute sourceUnit = ISQUnit::meter;
        attribute targetUnit = USCustomaryUnit::foot;
        attribute conversionFactor = 3.28084;
    }

    part massConverter : UnitConverter {
        attribute sourceUnit = ISQUnit::kilogram;
        attribute targetUnit = USCustomaryUnit::poundMass;
        attribute conversionFactor = 2.20462;
    }

    part temperatureConverter : UnitConverter {
        attribute sourceUnit = ISQUnit::kelvin;
        attribute targetUnit = USCustomaryUnit::rankine;
        attribute conversionFactor = 1.8;
        attribute offset = 0.0;
    }

    part fahrenheitConverter : UnitConverter {
        attribute sourceUnit = ISQUnit::kelvin;
        attribute targetUnit = USCustomaryUnit::fahrenheit;
        attribute conversionFactor = 1.8;
        attribute offset = 459.67;
    }
}