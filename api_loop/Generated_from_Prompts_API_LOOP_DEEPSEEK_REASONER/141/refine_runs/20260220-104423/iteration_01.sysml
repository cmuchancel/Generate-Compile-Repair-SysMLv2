package UnitSystemLibrary {
    
    private import ScalarValues::*;

    enum def ISQBaseUnits {
        enum meter;
        enum kilogram;
        enum second;
        enum ampere;
        enum kelvin;
        enum mole;
        enum candela;
    }

    enum def USCustomaryLengthUnits {
        enum inch;
        enum foot;
        enum yard;
        enum mile;
    }

    enum def USCustomaryMassUnits {
        enum pound;
        enum ounce;
        enum ton;
    }

    part def UnitConverter {
        attribute sourceValue : Real;
        attribute targetValue : Real;
        attribute conversionFactor : Real;
    }

    part def Quantity {
        attribute magnitude : Real;
        attribute unit : string;
    }

    part def LengthQuantity {
        attribute magnitude : Real;
        attribute unit : string;
    }

    part def MassQuantity {
        attribute magnitude : Real;
        attribute unit : string;
    }

    part def TimeQuantity {
        attribute magnitude : Real;
        attribute unit : string;
    }

    part def LengthConverter {
        part lengthUnitConverter : UnitConverter;
        attribute sourceUnit : USCustomaryLengthUnits;
        attribute targetUnit : ISQBaseUnits;
    }

    part def MassConverter {
        part massUnitConverter : UnitConverter;
        attribute sourceUnit : USCustomaryMassUnits;
        attribute targetUnit : ISQBaseUnits;
    }

    part def SystemController {
        part lengthConverter : LengthConverter;
        part massConverter : MassConverter;
        attribute activeConversions : Real;
    }

    part controller : SystemController;

    part converter1 : LengthConverter {
        part lengthUnitConverter : UnitConverter {
            attribute conversionFactor : Real = 0.0254;
        }
        attribute sourceUnit : USCustomaryLengthUnits = inch;
        attribute targetUnit : ISQBaseUnits = meter;
    }

    part converter2 : MassConverter {
        part massUnitConverter : UnitConverter {
            attribute conversionFactor : Real = 0.453592;
        }
        attribute sourceUnit : USCustomaryMassUnits = pound;
        attribute targetUnit : ISQBaseUnits = kilogram;
    }

    connect controller.lengthConverter to converter1;
    connect controller.massConverter to converter2;
}