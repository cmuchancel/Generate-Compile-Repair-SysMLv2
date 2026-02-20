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
        enum pound :> mass;
        enum ounce :> mass;
        enum second :> time;
        enum ampere :> electricCurrent;
        enum rankine :> thermodynamicTemperature;
        enum fahrenheit :> thermodynamicTemperature;
    }

    part def UnitConverter {
        attribute sourceUnit : String;
        attribute targetUnit : String;
        attribute conversionFactor : Real;
        attribute offset : Real = 0.0;

        port def input;
        port def output;

        port inputPort : input;
        port outputPort : output;
    }

    part def UnitSystem {
        attribute systemName : String;
        attribute systemDescription : String;

        attribute isqUnits : ISQUnit[0..*];
        attribute usCustomaryUnits : USCustomaryUnit[0..*];
        part converters : UnitConverter[0..*];
    }

    part standardizedUnitSystem : UnitSystem {
        attribute standardizedSystemName = "StandardizedUnitSystem";
        attribute standardizedSystemDescription = "Supports ISQ and US Customary Units interoperability";

        attribute isqUnitList : ISQUnit[0..*] = (ISQUnit::meter, ISQUnit::kilogram, ISQUnit::second, ISQUnit::ampere, ISQUnit::kelvin, ISQUnit::mole, ISQUnit::candela);
        attribute usCustomaryUnitList : USCustomaryUnit[0..*] = (USCustomaryUnit::inch, USCustomaryUnit::foot, USCustomaryUnit::yard, USCustomaryUnit::mile, USCustomaryUnit::pound, USCustomaryUnit::ounce, USCustomaryUnit::second, USCustomaryUnit::ampere, USCustomaryUnit::rankine, USCustomaryUnit::fahrenheit);

        part meterToInchConverter : UnitConverter {
            attribute sourceUnit = "ISQUnit::meter";
            attribute targetUnit = "USCustomaryUnit::inch";
            attribute conversionFactorValue = 39.3701;
        }

        part inchToMeterConverter : UnitConverter {
            attribute sourceUnit = "USCustomaryUnit::inch";
            attribute targetUnit = "ISQUnit::meter";
            attribute conversionFactorValue = 0.0254;
        }

        part kilogramToPoundConverter : UnitConverter {
            attribute sourceUnit = "ISQUnit::kilogram";
            attribute targetUnit = "USCustomaryUnit::pound";
            attribute conversionFactorValue = 2.20462;
        }

        part poundToKilogramConverter : UnitConverter {
            attribute sourceUnit = "USCustomaryUnit::pound";
            attribute targetUnit = "ISQUnit::kilogram";
            attribute conversionFactorValue = 0.453592;
        }

        part kelvinToRankineConverter : UnitConverter {
            attribute sourceUnit = "ISQUnit::kelvin";
            attribute targetUnit = "USCustomaryUnit::rankine";
            attribute conversionFactorValue = 1.8;
        }

        part rankineToKelvinConverter : UnitConverter {
            attribute sourceUnit = "USCustomaryUnit::rankine";
            attribute targetUnit = "ISQUnit::kelvin";
            attribute conversionFactorValue = 0.555556;
        }

        part kelvinToFahrenheitConverter : UnitConverter {
            attribute sourceUnit = "ISQUnit::kelvin";
            attribute targetUnit = "USCustomaryUnit::fahrenheit";
            attribute conversionFactorValue = 1.8;
            attribute offset = 459.67;
        }

        part fahrenheitToKelvinConverter : UnitConverter {
            attribute sourceUnit = "USCustomaryUnit::fahrenheit";
            attribute targetUnit = "ISQUnit::kelvin";
            attribute conversionFactorValue = 0.555556;
            attribute offset = -459.67;
        }
    }

    connect standardizedUnitSystem.meterToInchConverter.inputPort to standardizedUnitSystem.inchToMeterConverter.outputPort;
    connect standardizedUnitSystem.inchToMeterConverter.inputPort to standardizedUnitSystem.meterToInchConverter.outputPort;
    connect standardizedUnitSystem.kilogramToPoundConverter.inputPort to standardizedUnitSystem.poundToKilogramConverter.outputPort;
    connect standardizedUnitSystem.poundToKilogramConverter.inputPort to standardizedUnitSystem.kilogramToPoundConverter.outputPort;
    connect standardizedUnitSystem.kelvinToRankineConverter.inputPort to standardizedUnitSystem.rankineToKelvinConverter.outputPort;
    connect standardizedUnitSystem.rankineToKelvinConverter.inputPort to standardizedUnitSystem.kelvinToRankineConverter.outputPort;
}