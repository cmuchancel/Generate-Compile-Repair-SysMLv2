package UnitSystemLibrary {
private import ScalarValues::*;

enum def QuantityKind {
    enum Length;
    enum Mass;
    enum Time;
    enum Temperature;
    enum ElectricCurrent;
    enum AmountOfSubstance;
    enum LuminousIntensity;
    enum Area;
    enum Volume;
    enum Velocity;
    enum Acceleration;
    enum Force;
    enum Pressure;
    enum Energy;
    enum Power;
}

enum def UnitSystem {
    enum ISQ;
    enum USCustomary;
}

part def Unit {
    attribute unitName : String;
    attribute unitSymbol : String;
    attribute system : UnitSystem;
    attribute quantityKind : QuantityKind;
    attribute conversionFactorToSI : Real;
    attribute conversionOffsetToSI : Real;
}

part def ConversionRecord {
    attribute fromUnitName : String;
    attribute toUnitName : String;
    attribute factor : Real;
    attribute offset : Real;
}

part def ISQUnits {
    part metre : Unit;
    part kilogram : Unit;
    part second : Unit;
    part kelvin : Unit;
    part ampere : Unit;
    part mole : Unit;
    part candela : Unit;
    part squareMetre : Unit;
    part cubicMetre : Unit;
    part metrePerSecond : Unit;
    part metrePerSecondSquared : Unit;
    part newton : Unit;
    part pascal : Unit;
    part joule : Unit;
    part watt : Unit;
}

part def USCustomaryUnits {
    part foot : Unit;
    part inch : Unit;
    part yard : Unit;
    part mile : Unit;
    part poundMass : Unit;
    part ounce : Unit;
    part fahrenheit : Unit;
    part rankine : Unit;
    part squareFoot : Unit;
    part squareInch : Unit;
    part cubicFoot : Unit;
    part cubicInch : Unit;
    part footPerSecond : Unit;
    part milePerHour : Unit;
    part footPerSecondSquared : Unit;
    part poundForce : Unit;
    part poundPerSquareInch : Unit;
    part footPound : Unit;
    part horsepower : Unit;
}

part def UnitRegistry {
    part isqUnits : ISQUnits;
    part usCustomaryUnits : USCustomaryUnits;
}

part def ConversionTable {
    part lengthFtToM : ConversionRecord;
    part lengthInToM : ConversionRecord;
    part lengthYdToM : ConversionRecord;
    part lengthMiToM : ConversionRecord;
    part massPoundToKg : ConversionRecord;
    part massOunceToKg : ConversionRecord;
    part tempFToK : ConversionRecord;
    part tempRToK : ConversionRecord;
    part areaFt2ToM2 : ConversionRecord;
    part areaIn2ToM2 : ConversionRecord;
    part volumeFt3ToM3 : ConversionRecord;
    part volumeIn3ToM3 : ConversionRecord;
    part velocityFpsToMps : ConversionRecord;
    part velocityMphToMps : ConversionRecord;
    part accelFps2ToMps2 : ConversionRecord;
    part forceLbfToN : ConversionRecord;
    part pressurePsiToPa : ConversionRecord;
    part energyFtLbToJ : ConversionRecord;
    part powerHpToW : ConversionRecord;
}

part def MeasurementContext {
    attribute contextName : String;
    attribute defaultSystem : UnitSystem;
    part registry : UnitRegistry;
    part conversions : ConversionTable;
}

part engineeringContext : MeasurementContext;
part scienceContext : MeasurementContext;
part dailyLifeContext : MeasurementContext;

}