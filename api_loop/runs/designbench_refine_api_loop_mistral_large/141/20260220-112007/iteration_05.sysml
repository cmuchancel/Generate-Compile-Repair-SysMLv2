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
        attribute meterUnit : MeasurementUnit = new MeasurementUnit(symbol = "m", quantityKind = QuantityKind::length);
        attribute kilogramUnit : MeasurementUnit = new MeasurementUnit(symbol = "kg", quantityKind = QuantityKind::mass);
        attribute secondUnit : MeasurementUnit = new MeasurementUnit(symbol = "s", quantityKind = QuantityKind::time);
        attribute ampereUnit : MeasurementUnit = new MeasurementUnit(symbol = "A", quantityKind = QuantityKind::electricCurrent);
        attribute kelvinUnit : MeasurementUnit = new MeasurementUnit(symbol = "K", quantityKind = QuantityKind::thermodynamicTemperature);
        attribute moleUnit : MeasurementUnit = new MeasurementUnit(symbol = "mol", quantityKind = QuantityKind::amountOfSubstance);
        attribute candelaUnit : MeasurementUnit = new MeasurementUnit(symbol = "cd", quantityKind = QuantityKind::luminousIntensity);
        attribute radianUnit : MeasurementUnit = new MeasurementUnit(symbol = "rad", quantityKind = QuantityKind::planeAngle);
        attribute steradianUnit : MeasurementUnit = new MeasurementUnit(symbol = "sr", quantityKind = QuantityKind::solidAngle);
    }

    part usCustomaryUnits : USCustomaryUnit {
        attribute inchUnit : MeasurementUnit = new MeasurementUnit(symbol = "in", quantityKind = QuantityKind::length);
        attribute footUnit : MeasurementUnit = new MeasurementUnit(symbol = "ft", quantityKind = QuantityKind::length);
        attribute yardUnit : MeasurementUnit = new MeasurementUnit(symbol = "yd", quantityKind = QuantityKind::length);
        attribute mileUnit : MeasurementUnit = new MeasurementUnit(symbol = "mi", quantityKind = QuantityKind::length);
        attribute poundMassUnit : MeasurementUnit = new MeasurementUnit(symbol = "lbm", quantityKind = QuantityKind::mass);
        attribute slugUnit : MeasurementUnit = new MeasurementUnit(symbol = "slug", quantityKind = QuantityKind::mass);
        attribute secondUnit : MeasurementUnit = new MeasurementUnit(symbol = "s", quantityKind = QuantityKind::time);
        attribute ampereUnit : MeasurementUnit = new MeasurementUnit(symbol = "A", quantityKind = QuantityKind::electricCurrent);
        attribute rankineUnit : MeasurementUnit = new MeasurementUnit(symbol = "°R", quantityKind = QuantityKind::thermodynamicTemperature);
        attribute fahrenheitUnit : MeasurementUnit = new MeasurementUnit(symbol = "°F", quantityKind = QuantityKind::thermodynamicTemperature);
        attribute moleUnit : MeasurementUnit = new MeasurementUnit(symbol = "mol", quantityKind = QuantityKind::amountOfSubstance);
        attribute candelaUnit : MeasurementUnit = new MeasurementUnit(symbol = "cd", quantityKind = QuantityKind::luminousIntensity);
        attribute radianUnit : MeasurementUnit = new MeasurementUnit(symbol = "rad", quantityKind = QuantityKind::planeAngle);
        attribute steradianUnit : MeasurementUnit = new MeasurementUnit(symbol = "sr", quantityKind = QuantityKind::solidAngle);
    }

    part def UnitConverter {
        attribute sourceUnit : MeasurementUnit;
        attribute targetUnit : MeasurementUnit;
        attribute conversionFactor : Real;
        attribute offset : Real = 0.0;
    }

    part def UnitSystem {
        attribute systemName : String;
        attribute baseUnits : MeasurementUnit[0..*];
        attribute derivedUnits : MeasurementUnit[0..*];
    }

    part isqSystem : UnitSystem {
        attribute systemName = "International System of Quantities";
        attribute baseUnits = [isqUnits.meterUnit, isqUnits.kilogramUnit, isqUnits.secondUnit, isqUnits.ampereUnit, isqUnits.kelvinUnit, isqUnits.moleUnit, isqUnits.candelaUnit];
    }

    part usCustomarySystem : UnitSystem {
        attribute systemName = "United States Customary Units";
        attribute baseUnits = [usCustomaryUnits.footUnit, usCustomaryUnits.poundMassUnit, usCustomaryUnits.secondUnit, usCustomaryUnits.ampereUnit];
    }

    part def Quantity {
        attribute value : Real;
        attribute unit : MeasurementUnit;
    }

    part lengthConverter : UnitConverter {
        attribute sourceUnit = isqUnits.meterUnit;
        attribute targetUnit = usCustomaryUnits.footUnit;
        attribute conversionFactor = 3.28084;
    }

    part massConverter : UnitConverter {
        attribute sourceUnit = isqUnits.kilogramUnit;
        attribute targetUnit = usCustomaryUnits.poundMassUnit;
        attribute conversionFactor = 2.20462;
    }

    part temperatureConverter : UnitConverter {
        attribute sourceUnit = isqUnits.kelvinUnit;
        attribute targetUnit = usCustomaryUnits.rankineUnit;
        attribute conversionFactor = 1.8;
        attribute offset = 0.0;
    }

    part temperatureOffsetConverter : UnitConverter {
        attribute sourceUnit = isqUnits.kelvinUnit;
        attribute targetUnit = usCustomaryUnits.fahrenheitUnit;
        attribute conversionFactor = 1.8;
        attribute offset = 459.67;
    }
}