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
        enum frequency;
        enum force;
        enum pressure;
        enum energy;
        enum power;
        enum electricCharge;
        enum voltage;
        enum capacitance;
        enum electricResistance;
        enum electricConductance;
        enum magneticFlux;
        enum magneticFluxDensity;
        enum inductance;
        enum temperature;
        enum area;
        enum volume;
        enum velocity;
        enum acceleration;
        enum angularVelocity;
        enum angularAcceleration;
        enum density;
        enum surfaceDensity;
        enum specificVolume;
        enum luminance;
    }

    enum def UnitSystem {
        enum ISQ;
        enum USCustomaryUnits;
    }

    part def Unit {
        attribute name : String;
        attribute symbol : String;
        attribute quantityKind : QuantityKind;
        attribute conversionFactorToBase : Real;
        attribute offsetToBase : Real = 0.0;
        attribute unitSystem : UnitSystem;
    }

    part def UnitConverter {
        attribute sourceUnit : Unit;
        attribute targetUnit : Unit;
        attribute conversionFactor : Real;
        attribute offset : Real = 0.0;

        port def convert : Real;
    }

    part def UnitLibrary {
        attribute units : Unit[0..*];
        attribute converters : UnitConverter[0..*];

        port def addUnit : Unit;
        port def removeUnit : Unit;
        port def findUnit : Unit;
        port def convertValue : Real;
    }

    part isqLibrary : UnitLibrary {
        attribute units = (
            part meter : Unit {
                attribute name = "meter";
                attribute symbol = "m";
                attribute quantityKind = QuantityKind::length;
                attribute conversionFactorToBase = 1.0;
                attribute unitSystem = UnitSystem::ISQ;
            },
            part kilogram : Unit {
                attribute name = "kilogram";
                attribute symbol = "kg";
                attribute quantityKind = QuantityKind::mass;
                attribute conversionFactorToBase = 1.0;
                attribute unitSystem = UnitSystem::ISQ;
            },
            part second : Unit {
                attribute name = "second";
                attribute symbol = "s";
                attribute quantityKind = QuantityKind::time;
                attribute conversionFactorToBase = 1.0;
                attribute unitSystem = UnitSystem::ISQ;
            },
            part ampere : Unit {
                attribute name = "ampere";
                attribute symbol = "A";
                attribute quantityKind = QuantityKind::electricCurrent;
                attribute conversionFactorToBase = 1.0;
                attribute unitSystem = UnitSystem::ISQ;
            },
            part kelvin : Unit {
                attribute name = "kelvin";
                attribute symbol = "K";
                attribute quantityKind = QuantityKind::thermodynamicTemperature;
                attribute conversionFactorToBase = 1.0;
                attribute unitSystem = UnitSystem::ISQ;
            },
            part mole : Unit {
                attribute name = "mole";
                attribute symbol = "mol";
                attribute quantityKind = QuantityKind::amountOfSubstance;
                attribute conversionFactorToBase = 1.0;
                attribute unitSystem = UnitSystem::ISQ;
            },
            part candela : Unit {
                attribute name = "candela";
                attribute symbol = "cd";
                attribute quantityKind = QuantityKind::luminousIntensity;
                attribute conversionFactorToBase = 1.0;
                attribute unitSystem = UnitSystem::ISQ;
            }
        );
    }

    part usCustomaryLibrary : UnitLibrary {
        attribute units = (
            part inch : Unit {
                attribute name = "inch";
                attribute symbol = "in";
                attribute quantityKind = QuantityKind::length;
                attribute conversionFactorToBase = 0.0254;
                attribute unitSystem = UnitSystem::USCustomaryUnits;
            },
            part foot : Unit {
                attribute name = "foot";
                attribute symbol = "ft";
                attribute quantityKind = QuantityKind::length;
                attribute conversionFactorToBase = 0.3048;
                attribute unitSystem = UnitSystem::USCustomaryUnits;
            },
            part yard : Unit {
                attribute name = "yard";
                attribute symbol = "yd";
                attribute quantityKind = QuantityKind::length;
                attribute conversionFactorToBase = 0.9144;
                attribute unitSystem = UnitSystem::USCustomaryUnits;
            },
            part mile : Unit {
                attribute name = "mile";
                attribute symbol = "mi";
                attribute quantityKind = QuantityKind::length;
                attribute conversionFactorToBase = 1609.344;
                attribute unitSystem = UnitSystem::USCustomaryUnits;
            },
            part poundMass : Unit {
                attribute name = "pound-mass";
                attribute symbol = "lbm";
                attribute quantityKind = QuantityKind::mass;
                attribute conversionFactorToBase = 0.45359237;
                attribute unitSystem = UnitSystem::USCustomaryUnits;
            },
            part ounce : Unit {
                attribute name = "ounce";
                attribute symbol = "oz";
                attribute quantityKind = QuantityKind::mass;
                attribute conversionFactorToBase = 0.028349523125;
                attribute unitSystem = UnitSystem::USCustomaryUnits;
            },
            part second : Unit {
                attribute name = "second";
                attribute symbol = "s";
                attribute quantityKind = QuantityKind::time;
                attribute conversionFactorToBase = 1.0;
                attribute unitSystem = UnitSystem::USCustomaryUnits;
            }
        );
    }

    part converter : UnitConverter {
        attribute sourceUnit = usCustomaryLibrary::units[0]; // inch
        attribute targetUnit = isqLibrary::units[0]; // meter
        attribute conversionFactor = 0.0254;
    }

    connect usCustomaryLibrary::convertValue to converter::convert;
    connect converter::convert to isqLibrary::convertValue;
}