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
        enum electricPotential;
        enum electricCapacitance;
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
        enum USCustomary;
    }

    part def Unit {
        attribute name : String;
        attribute symbol : String;
        attribute quantityKind : QuantityKind;
        attribute conversionFactorToBase : Real;
        attribute system : UnitSystem;
    }

    part def UnitConverter {
        attribute sourceUnit : Unit;
        attribute targetUnit : Unit;
        attribute conversionFactor : Real;

        port def inputPort;
        port def outputPort;

        port input : inputPort;
        port output : outputPort;
    }

    part def UnitLibrary {
        attribute units : Unit[*];
        attribute converters : UnitConverter[*];

        part isqUnits : Unit[*] {
            attribute :>> units;
        }

        part usCustomaryUnits : Unit[*] {
            attribute :>> units;
        }

        part isqToUSConverters : UnitConverter[*] {
            attribute :>> converters;
        }

        part usToISQConverters : UnitConverter[*] {
            attribute :>> converters;
        }
    }

    part unitLibrary : UnitLibrary;

    connect unitLibrary.isqToUSConverters->input to unitLibrary.isqUnits;
    connect unitLibrary.usToISQConverters->input to unitLibrary.usCustomaryUnits;
    connect unitLibrary.isqToUSConverters->output to unitLibrary.usCustomaryUnits;
    connect unitLibrary.usToISQConverters->output to unitLibrary.isqUnits;
}