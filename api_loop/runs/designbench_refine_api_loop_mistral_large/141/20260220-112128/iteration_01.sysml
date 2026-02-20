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
        attribute sourceUnit : ISQUnit | USCustomaryUnit;
        attribute targetUnit : ISQUnit | USCustomaryUnit;
        attribute conversionFactor : Real;
        attribute offset : Real = 0.0;

        port def input;
        port def output;

        port inputPort : input;
        port outputPort : output;
    }

    part def UnitSystem {
        attribute name : String;
        attribute description : String;

        part isqUnits : ISQUnit[0..*];
        part usCustomaryUnits : USCustomaryUnit[0..*];
        part converters : UnitConverter[0..*];
    }

    part unitSystem : UnitSystem {
        attribute name = "StandardizedUnitSystem";
        attribute description = "Supports ISQ and US Customary Units interoperability";

        part isqUnits : ISQUnit = (meter, kilogram, second, ampere, kelvin, mole, candela);
        part usCustomaryUnits : USCustomaryUnit = (inch, foot, yard, mile, pound, ounce, second, ampere, rankine, fahrenheit);

        part meterToInch : UnitConverter {
            attribute sourceUnit = ISQUnit::meter;
            attribute targetUnit = USCustomaryUnit::inch;
            attribute conversionFactor = 39.3701;
        }

        part inchToMeter : UnitConverter {
            attribute sourceUnit = USCustomaryUnit::inch;
            attribute targetUnit = ISQUnit::meter;
            attribute conversionFactor = 0.0254;
        }

        part kilogramToPound : UnitConverter {
            attribute sourceUnit = ISQUnit::kilogram;
            attribute targetUnit = USCustomaryUnit::pound;
            attribute conversionFactor = 2.20462;
        }

        part poundToKilogram : UnitConverter {
            attribute sourceUnit = USCustomaryUnit::pound;
            attribute targetUnit = ISQUnit::kilogram;
            attribute conversionFactor = 0.453592;
        }

        part kelvinToRankine : UnitConverter {
            attribute sourceUnit = ISQUnit::kelvin;
            attribute targetUnit = USCustomaryUnit::rankine;
            attribute conversionFactor = 1.8;
        }

        part rankineToKelvin : UnitConverter {
            attribute sourceUnit = USCustomaryUnit::rankine;
            attribute targetUnit = ISQUnit::kelvin;
            attribute conversionFactor = 0.555556;
        }

        part celsiusToFahrenheit : UnitConverter {
            attribute sourceUnit = ISQUnit::kelvin;
            attribute targetUnit = USCustomaryUnit::fahrenheit;
            attribute conversionFactor = 1.8;
            attribute offset = 32.0;
        }

        part fahrenheitToCelsius : UnitConverter {
            attribute sourceUnit = USCustomaryUnit::fahrenheit;
            attribute targetUnit = ISQUnit::kelvin;
            attribute conversionFactor = 0.555556;
            attribute offset = -32.0;
        }
    }

    connect unitSystem.meterToInch.inputPort to unitSystem.inchToMeter.outputPort;
    connect unitSystem.inchToMeter.inputPort to unitSystem.meterToInch.outputPort;
    connect unitSystem.kilogramToPound.inputPort to unitSystem.poundToKilogram.outputPort;
    connect unitSystem.poundToKilogram.inputPort to unitSystem.kilogramToPound.outputPort;
    connect unitSystem.kelvinToRankine.inputPort to unitSystem.rankineToKelvin.outputPort;
    connect unitSystem.rankineToKelvin.inputPort to unitSystem.kelvinToRankine.outputPort;
}