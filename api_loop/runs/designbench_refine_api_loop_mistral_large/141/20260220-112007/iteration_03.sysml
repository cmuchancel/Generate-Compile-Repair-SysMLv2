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
        enum planeAngle;
        enum solidAngle;
        enum area;
        enum volume;
        enum velocity;
        enum acceleration;
        enum force;
        enum pressure;
        enum energy;
        enum power;
        enum electricCharge;
        enum electricPotential;
        enum electricResistance;
        enum frequency;
    }

    part def MeasurementUnit {
        attribute symbol : String;
        attribute quantityKind : QuantityKind;
    }

    part def ISQUnit {
        attribute meter : MeasurementUnit;
        attribute kilogram : MeasurementUnit;
        attribute second : MeasurementUnit;
        attribute ampere : MeasurementUnit;
        attribute kelvin : MeasurementUnit;
        attribute mole : MeasurementUnit;
        attribute candela : MeasurementUnit;
        attribute radian : MeasurementUnit;
        attribute steradian : MeasurementUnit;
    }

    part def USCustomaryUnit {
        attribute inch : MeasurementUnit;
        attribute foot : MeasurementUnit;
        attribute yard : MeasurementUnit;
        attribute mile : MeasurementUnit;
        attribute poundMass : MeasurementUnit;
        attribute slug : MeasurementUnit;
        attribute second : MeasurementUnit;
        attribute ampere : MeasurementUnit;
        attribute rankine : MeasurementUnit;
        attribute fahrenheit : MeasurementUnit;
        attribute mole : MeasurementUnit;
        attribute candela : MeasurementUnit;
        attribute radian : MeasurementUnit;
        attribute steradian : MeasurementUnit;
    }

    part isqUnits : ISQUnit {
        attribute meter = (symbol = "m", quantityKind = QuantityKind::length);
        attribute kilogram = (symbol = "kg", quantityKind = QuantityKind::mass);
        attribute second = (symbol = "s", quantityKind = QuantityKind::time);
        attribute ampere = (symbol = "A", quantityKind = QuantityKind::electricCurrent);
        attribute kelvin = (symbol = "K", quantityKind = QuantityKind::thermodynamicTemperature);
        attribute mole = (symbol = "mol", quantityKind = QuantityKind::amountOfSubstance);
        attribute candela = (symbol = "cd", quantityKind = QuantityKind::luminousIntensity);
        attribute radian = (symbol = "rad", quantityKind = QuantityKind::planeAngle);
        attribute steradian = (symbol = "sr", quantityKind = QuantityKind::solidAngle);
    }

    part usCustomaryUnits : USCustomaryUnit {
        attribute inch = (symbol = "in", quantityKind = QuantityKind::length);
        attribute foot = (symbol = "ft", quantityKind = QuantityKind::length);
        attribute yard = (symbol = "yd", quantityKind = QuantityKind::length);
        attribute mile = (symbol = "mi", quantityKind = QuantityKind::length);
        attribute poundMass = (symbol = "lbm", quantityKind = QuantityKind::mass);
        attribute slug = (symbol = "slug", quantityKind = QuantityKind::mass);
        attribute second = (symbol = "s", quantityKind = QuantityKind::time);
        attribute ampere = (symbol = "A", quantityKind = QuantityKind::electricCurrent);
        attribute rankine = (symbol = "°R", quantityKind = QuantityKind::thermodynamicTemperature);
        attribute fahrenheit = (symbol = "°F", quantityKind = QuantityKind::thermodynamicTemperature);
        attribute mole = (symbol = "mol", quantityKind = QuantityKind::amountOfSubstance);
        attribute candela = (symbol = "cd", quantityKind = QuantityKind::luminousIntensity);
        attribute radian = (symbol = "rad", quantityKind = QuantityKind::planeAngle);
        attribute steradian = (symbol = "sr", quantityKind = QuantityKind::solidAngle);
    }

    part def UnitConverter {
        attribute sourceUnit : MeasurementUnit;
        attribute targetUnit : MeasurementUnit;
        attribute conversionFactor : Real;
        attribute offset : Real = 0.0;
    }

    part def UnitSystem {
        attribute systemName : String;
        attribute baseUnits : MeasurementUnit[*];
        attribute derivedUnits : MeasurementUnit[*];
    }

    part isqSystem : UnitSystem {
        attribute systemName = "International System of Quantities";
        attribute baseUnits = [isqUnits.meter, isqUnits.kilogram, isqUnits.second, isqUnits.ampere, isqUnits.kelvin, isqUnits.mole, isqUnits.candela];
    }

    part usCustomarySystem : UnitSystem {
        attribute systemName = "United States Customary Units";
        attribute baseUnits = [usCustomaryUnits.foot, usCustomaryUnits.poundMass, usCustomaryUnits.second, usCustomaryUnits.ampere];
    }

    part def Quantity {
        attribute value : Real;
        attribute unit : MeasurementUnit;
    }

    part lengthConverter : UnitConverter {
        attribute sourceUnit = isqUnits.meter;
        attribute targetUnit = usCustomaryUnits.foot;
        attribute conversionFactor = 3.28084;
    }

    part massConverter : UnitConverter {
        attribute sourceUnit = isqUnits.kilogram;
        attribute targetUnit = usCustomaryUnits.poundMass;
        attribute conversionFactor = 2.20462;
    }

    part temperatureConverter : UnitConverter {
        attribute sourceUnit = isqUnits.kelvin;
        attribute targetUnit = usCustomaryUnits.rankine;
        attribute conversionFactor = 1.8;
        attribute offset = 0.0;
    }

    part temperatureOffsetConverter : UnitConverter {
        attribute sourceUnit = isqUnits.kelvin;
        attribute targetUnit = usCustomaryUnits.fahrenheit;
        attribute conversionFactor = 1.8;
        attribute offset = 459.67;
    }
}