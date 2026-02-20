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
        part metre : Unit {
            attribute unitName : String = "metre";
            attribute unitSymbol : String = "m";
            attribute system : UnitSystem = UnitSystem::ISQ;
            attribute quantityKind : QuantityKind = QuantityKind::Length;
            attribute conversionFactorToSI : Real = 1.0;
            attribute conversionOffsetToSI : Real = 0.0;
        }
        part kilogram : Unit {
            attribute unitName : String = "kilogram";
            attribute unitSymbol : String = "kg";
            attribute system : UnitSystem = UnitSystem::ISQ;
            attribute quantityKind : QuantityKind = QuantityKind::Mass;
            attribute conversionFactorToSI : Real = 1.0;
            attribute conversionOffsetToSI : Real = 0.0;
        }
        part second : Unit {
            attribute unitName : String = "second";
            attribute unitSymbol : String = "s";
            attribute system : UnitSystem = UnitSystem::ISQ;
            attribute quantityKind : QuantityKind = QuantityKind::Time;
            attribute conversionFactorToSI : Real = 1.0;
            attribute conversionOffsetToSI : Real = 0.0;
        }
        part kelvin : Unit {
            attribute unitName : String = "kelvin";
            attribute unitSymbol : String = "K";
            attribute system : UnitSystem = UnitSystem::ISQ;
            attribute quantityKind : QuantityKind = QuantityKind::Temperature;
            attribute conversionFactorToSI : Real = 1.0;
            attribute conversionOffsetToSI : Real = 0.0;
        }
        part ampere : Unit {
            attribute unitName : String = "ampere";
            attribute unitSymbol : String = "A";
            attribute system : UnitSystem = UnitSystem::ISQ;
            attribute quantityKind : QuantityKind = QuantityKind::ElectricCurrent;
            attribute conversionFactorToSI : Real = 1.0;
            attribute conversionOffsetToSI : Real = 0.0;
        }
        part mole : Unit {
            attribute unitName : String = "mole";
            attribute unitSymbol : String = "mol";
            attribute system : UnitSystem = UnitSystem::ISQ;
            attribute quantityKind : QuantityKind = QuantityKind::AmountOfSubstance;
            attribute conversionFactorToSI : Real = 1.0;
            attribute conversionOffsetToSI : Real = 0.0;
        }
        part candela : Unit {
            attribute unitName : String = "candela";
            attribute unitSymbol : String = "cd";
            attribute system : UnitSystem = UnitSystem::ISQ;
            attribute quantityKind : QuantityKind = QuantityKind::LuminousIntensity;
            attribute conversionFactorToSI : Real = 1.0;
            attribute conversionOffsetToSI : Real = 0.0;
        }
        part squareMetre : Unit {
            attribute unitName : String = "square metre";
            attribute unitSymbol : String = "m2";
            attribute system : UnitSystem = UnitSystem::ISQ;
            attribute quantityKind : QuantityKind = QuantityKind::Area;
            attribute conversionFactorToSI : Real = 1.0;
            attribute conversionOffsetToSI : Real = 0.0;
        }
        part cubicMetre : Unit {
            attribute unitName : String = "cubic metre";
            attribute unitSymbol : String = "m3";
            attribute system : UnitSystem = UnitSystem::ISQ;
            attribute quantityKind : QuantityKind = QuantityKind::Volume;
            attribute conversionFactorToSI : Real = 1.0;
            attribute conversionOffsetToSI : Real = 0.0;
        }
        part metrePerSecond : Unit {
            attribute unitName : String = "metre per second";
            attribute unitSymbol : String = "m/s";
            attribute system : UnitSystem = UnitSystem::ISQ;
            attribute quantityKind : QuantityKind = QuantityKind::Velocity;
            attribute conversionFactorToSI : Real = 1.0;
            attribute conversionOffsetToSI : Real = 0.0;
        }
        part metrePerSecondSquared : Unit {
            attribute unitName : String = "metre per second squared";
            attribute unitSymbol : String = "m/s2";
            attribute system : UnitSystem = UnitSystem::ISQ;
            attribute quantityKind : QuantityKind = QuantityKind::Acceleration;
            attribute conversionFactorToSI : Real = 1.0;
            attribute conversionOffsetToSI : Real = 0.0;
        }
        part newton : Unit {
            attribute unitName : String = "newton";
            attribute unitSymbol : String = "N";
            attribute system : UnitSystem = UnitSystem::ISQ;
            attribute quantityKind : QuantityKind = QuantityKind::Force;
            attribute conversionFactorToSI : Real = 1.0;
            attribute conversionOffsetToSI : Real = 0.0;
        }
        part pascal : Unit {
            attribute unitName : String = "pascal";
            attribute unitSymbol : String = "Pa";
            attribute system : UnitSystem = UnitSystem::ISQ;
            attribute quantityKind : QuantityKind = QuantityKind::Pressure;
            attribute conversionFactorToSI : Real = 1.0;
            attribute conversionOffsetToSI : Real = 0.0;
        }
        part joule : Unit {
            attribute unitName : String = "joule";
            attribute unitSymbol : String = "J";
            attribute system : UnitSystem = UnitSystem::ISQ;
            attribute quantityKind : QuantityKind = QuantityKind::Energy;
            attribute conversionFactorToSI : Real = 1.0;
            attribute conversionOffsetToSI : Real = 0.0;
        }
        part watt : Unit {
            attribute unitName : String = "watt";
            attribute unitSymbol : String = "W";
            attribute system : UnitSystem = UnitSystem::ISQ;
            attribute quantityKind : QuantityKind = QuantityKind::Power;
            attribute conversionFactorToSI : Real = 1.0;
            attribute conversionOffsetToSI : Real = 0.0;
        }
    }

    part def USCustomaryUnits {
        part foot : Unit {
            attribute unitName : String = "foot";
            attribute unitSymbol : String = "ft";
            attribute system : UnitSystem = UnitSystem::USCustomary;
            attribute quantityKind : QuantityKind = QuantityKind::Length;
            attribute conversionFactorToSI : Real = 0.3048;
            attribute conversionOffsetToSI : Real = 0.0;
        }
        part inch : Unit {
            attribute unitName : String = "inch";
            attribute unitSymbol : String = "in";
            attribute system : UnitSystem = UnitSystem::USCustomary;
            attribute quantityKind : QuantityKind = QuantityKind::Length;
            attribute conversionFactorToSI : Real = 0.0254;
            attribute conversionOffsetToSI : Real = 0.0;
        }
        part yard : Unit {
            attribute unitName : String = "yard";
            attribute unitSymbol : String = "yd";
            attribute system : UnitSystem = UnitSystem::USCustomary;
            attribute quantityKind : QuantityKind = QuantityKind::Length;
            attribute conversionFactorToSI : Real = 0.9144;
            attribute conversionOffsetToSI : Real = 0.0;
        }
        part mile : Unit {
            attribute unitName : String = "mile";
            attribute unitSymbol : String = "mi";
            attribute system : UnitSystem = UnitSystem::USCustomary;
            attribute quantityKind : QuantityKind = QuantityKind::Length;
            attribute conversionFactorToSI : Real = 1609.344;
            attribute conversionOffsetToSI : Real = 0.0;
        }
        part poundMass : Unit {
            attribute unitName : String = "pound mass";
            attribute unitSymbol : String = "lbm";
            attribute system : UnitSystem = UnitSystem::USCustomary;
            attribute quantityKind : QuantityKind = QuantityKind::Mass;
            attribute conversionFactorToSI : Real = 0.45359237;
            attribute conversionOffsetToSI : Real = 0.0;
        }
        part ounce : Unit {
            attribute unitName : String = "ounce";
            attribute unitSymbol : String = "oz";
            attribute system : UnitSystem = UnitSystem::USCustomary;
            attribute quantityKind : QuantityKind = QuantityKind::Mass;
            attribute conversionFactorToSI : Real = 0.028349523;
            attribute conversionOffsetToSI : Real = 0.0;
        }
        part fahrenheit : Unit {
            attribute unitName : String = "fahrenheit";
            attribute unitSymbol : String = "degF";
            attribute system : UnitSystem = UnitSystem::USCustomary;
            attribute quantityKind : QuantityKind = QuantityKind::Temperature;
            attribute conversionFactorToSI : Real = 0.5555556;
            attribute conversionOffsetToSI : Real = 255.3722;
        }
        part rankine : Unit {
            attribute unitName : String = "rankine";
            attribute unitSymbol : String = "degR";
            attribute system : UnitSystem = UnitSystem::USCustomary;
            attribute quantityKind : QuantityKind = QuantityKind::Temperature;
            attribute conversionFactorToSI : Real = 0.5555556;
            attribute conversionOffsetToSI : Real = 0.0;
        }
        part squareFoot : Unit {
            attribute unitName : String = "square foot";
            attribute unitSymbol : String = "ft2";
            attribute system : UnitSystem = UnitSystem::USCustomary;
            attribute quantityKind : QuantityKind = QuantityKind::Area;
            attribute conversionFactorToSI : Real = 0.09290304;
            attribute conversionOffsetToSI : Real = 0.0;
        }
        part squareInch : Unit {
            attribute unitName : String = "square inch";
            attribute unitSymbol : String = "in2";