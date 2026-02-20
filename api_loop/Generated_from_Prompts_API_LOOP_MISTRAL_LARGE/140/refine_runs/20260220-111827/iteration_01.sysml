package PhysicalQuantities {
    private import ScalarValues::*;

    enum def BaseQuantityKind {
        enum length;
        enum mass;
        enum time;
        enum electricCurrent;
        enum thermodynamicTemperature;
        enum amountOfSubstance;
        enum luminousIntensity;
    }

    part def PhysicalQuantity {
        attribute name : String;
        attribute symbol : String;
        attribute baseQuantity : BaseQuantityKind;
        attribute dimension : String;
    }

    part def Unit {
        attribute name : String;
        attribute symbol : String;
        attribute conversionFactor : Real;
        attribute offset : Real = 0.0;
        attribute quantityKind : BaseQuantityKind;
    }

    part def QuantitySystem {
        part quantities : PhysicalQuantity[*];
        part units : Unit[*];

        attribute name : String;

        port def QuantityPort {
            attribute quantity : PhysicalQuantity;
        }

        port quantityInput : QuantityPort;
        port quantityOutput : QuantityPort;
    }

    part def UnitConverter {
        attribute sourceUnit : Unit;
        attribute targetUnit : Unit;

        port def ConversionPort {
            attribute value : Real;
        }

        port input : ConversionPort;
        port output : ConversionPort;

        connect input to output;
    }

    part isq : QuantitySystem {
        attribute name : String = "International System of Quantities";

        part length : PhysicalQuantity {
            attribute name : String = "length";
            attribute symbol : String = "l";
            attribute baseQuantity : BaseQuantityKind = BaseQuantityKind::length;
            attribute dimension : String = "L";
        }

        part mass : PhysicalQuantity {
            attribute name : String = "mass";
            attribute symbol : String = "m";
            attribute baseQuantity : BaseQuantityKind = BaseQuantityKind::mass;
            attribute dimension : String = "M";
        }

        part time : PhysicalQuantity {
            attribute name : String = "time";
            attribute symbol : String = "t";
            attribute baseQuantity : BaseQuantityKind = BaseQuantityKind::time;
            attribute dimension : String = "T";
        }

        part electricCurrent : PhysicalQuantity {
            attribute name : String = "electric current";
            attribute symbol : String = "I";
            attribute baseQuantity : BaseQuantityKind = BaseQuantityKind::electricCurrent;
            attribute dimension : String = "I";
        }

        part thermodynamicTemperature : PhysicalQuantity {
            attribute name : String = "thermodynamic temperature";
            attribute symbol : String = "Θ";
            attribute baseQuantity : BaseQuantityKind = BaseQuantityKind::thermodynamicTemperature;
            attribute dimension : String = "Θ";
        }

        part amountOfSubstance : PhysicalQuantity {
            attribute name : String = "amount of substance";
            attribute symbol : String = "n";
            attribute baseQuantity : BaseQuantityKind = BaseQuantityKind::amountOfSubstance;
            attribute dimension : String = "N";
        }

        part luminousIntensity : PhysicalQuantity {
            attribute name : String = "luminous intensity";
            attribute symbol : String = "J";
            attribute baseQuantity : BaseQuantityKind = BaseQuantityKind::luminousIntensity;
            attribute dimension : String = "J";
        }

        part meter : Unit {
            attribute name : String = "meter";
            attribute symbol : String = "m";
            attribute conversionFactor : Real = 1.0;
            attribute quantityKind : BaseQuantityKind = BaseQuantityKind::length;
        }

        part kilogram : Unit {
            attribute name : String = "kilogram";
            attribute symbol : String = "kg";
            attribute conversionFactor : Real = 1.0;
            attribute quantityKind : BaseQuantityKind = BaseQuantityKind::mass;
        }

        part second : Unit {
            attribute name : String = "second";
            attribute symbol : String = "s";
            attribute conversionFactor : Real = 1.0;
            attribute quantityKind : BaseQuantityKind = BaseQuantityKind::time;
        }

        part ampere : Unit {
            attribute name : String = "ampere";
            attribute symbol : String = "A";
            attribute conversionFactor : Real = 1.0;
            attribute quantityKind : BaseQuantityKind = BaseQuantityKind::electricCurrent;
        }

        part kelvin : Unit {
            attribute name : String = "kelvin";
            attribute symbol : String = "K";
            attribute conversionFactor : Real = 1.0;
            attribute quantityKind : BaseQuantityKind = BaseQuantityKind::thermodynamicTemperature;
        }

        part mole : Unit {
            attribute name : String = "mole";
            attribute symbol : String = "mol";
            attribute conversionFactor : Real = 1.0;
            attribute quantityKind : BaseQuantityKind = BaseQuantityKind::amountOfSubstance;
        }

        part candela : Unit {
            attribute name : String = "candela";
            attribute symbol : String = "cd";
            attribute conversionFactor : Real = 1.0;
            attribute quantityKind : BaseQuantityKind = BaseQuantityKind::luminousIntensity;
        }
    }
}