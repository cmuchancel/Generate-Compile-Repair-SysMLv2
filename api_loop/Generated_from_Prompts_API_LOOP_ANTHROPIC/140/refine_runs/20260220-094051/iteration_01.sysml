package ISQPhysicalQuantities {
    private import ScalarValues::*;

    enum def PhysicalQuantityKind {
        enum length;
        enum mass;
        enum time;
        enum electricCurrent;
        enum temperature;
        enum amountOfSubstance;
        enum luminousIntensity;
    }

    part def PhysicalQuantity {
        attribute quantityKind : PhysicalQuantityKind;
        attribute magnitude : Real;
        attribute unitSymbol : String;
        attribute unitName : String;
    }

    part def LengthQuantity :> PhysicalQuantity {
        attribute redefines quantityKind = PhysicalQuantityKind::length;
        attribute redefines unitSymbol = "m";
        attribute redefines unitName = "metre";
    }

    part def MassQuantity :> PhysicalQuantity {
        attribute redefines quantityKind = PhysicalQuantityKind::mass;
        attribute redefines unitSymbol = "kg";
        attribute redefines unitName = "kilogram";
    }

    part def TimeQuantity :> PhysicalQuantity {
        attribute redefines quantityKind = PhysicalQuantityKind::time;
        attribute redefines unitSymbol = "s";
        attribute redefines unitName = "second";
    }

    part def ElectricCurrentQuantity :> PhysicalQuantity {
        attribute redefines quantityKind = PhysicalQuantityKind::electricCurrent;
        attribute redefines unitSymbol = "A";
        attribute redefines unitName = "ampere";
    }

    part def TemperatureQuantity :> PhysicalQuantity {
        attribute redefines quantityKind = PhysicalQuantityKind::temperature;
        attribute redefines unitSymbol = "K";
        attribute redefines unitName = "kelvin";
    }

    part def AmountOfSubstanceQuantity :> PhysicalQuantity {
        attribute redefines quantityKind = PhysicalQuantityKind::amountOfSubstance;
        attribute redefines unitSymbol = "mol";
        attribute redefines unitName = "mole";
    }

    part def LuminousIntensityQuantity :> PhysicalQuantity {
        attribute redefines quantityKind = PhysicalQuantityKind::luminousIntensity;
        attribute redefines unitSymbol = "cd";
        attribute redefines unitName = "candela";
    }

    part def ISQSystem {
        part lengthDim : LengthQuantity;
        part massDim : MassQuantity;
        part timeDim : TimeQuantity;
        part electricCurrentDim : ElectricCurrentQuantity;
        part temperatureDim : TemperatureQuantity;
        part amountOfSubstanceDim : AmountOfSubstanceQuantity;
        part luminousIntensityDim : LuminousIntensityQuantity;
    }

    part def UnitConversionRecord {
        attribute fromUnit : String;
        attribute toUnit : String;
        attribute conversionFactor : Real;
        attribute offset : Real;
    }

    part def LengthConversionRecord :> UnitConversionRecord {
        attribute redefines fromUnit = "m";
        attribute redefines toUnit = "cm";
        attribute redefines conversionFactor = 100.0;
        attribute redefines offset = 0.0;
    }

    part def MassConversionRecord :> UnitConversionRecord {
        attribute redefines fromUnit = "kg";
        attribute redefines toUnit = "g";
        attribute redefines conversionFactor = 1000.0;
        attribute redefines offset = 0.0;
    }

    part def TemperatureConversionRecord :> UnitConversionRecord {
        attribute redefines fromUnit = "K";
        attribute redefines toUnit = "C";
        attribute redefines conversionFactor = 1.0;
        attribute redefines offset = -273.15;
    }

    part def ISQDataManager {
        part isqSystem : ISQSystem;
        part lengthConversion : LengthConversionRecord;
        part massConversion : MassConversionRecord;
        part temperatureConversion : TemperatureConversionRecord;
        attribute systemName : String;
        attribute systemVersion : String;
    }

    part isqInstance : ISQDataManager {
        attribute systemName = "ISQ Physical Quantity System";
        attribute systemVersion = "1.0";
    }
}