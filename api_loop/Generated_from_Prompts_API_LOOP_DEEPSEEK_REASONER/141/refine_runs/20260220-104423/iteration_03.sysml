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
        attribute factor : Real;
    }

    part def Quantity {
        attribute magnitude : Real;
        attribute unitName : String;
    }

    part def LengthQuantity {
        part baseQuantity : Quantity;
    }

    part def MassQuantity {
        part baseQuantity : Quantity;
    }

    part def TimeQuantity {
        part baseQuantity : Quantity;
    }

    part def LengthConverter {
        part myUnitConverter : UnitConverter;
        attribute fromUnit : USCustomaryLengthUnits;
        attribute toUnit : ISQBaseUnits;
    }

    part def MassConverter {
        part myUnitConverter : UnitConverter;
        attribute fromUnit : USCustomaryMassUnits;
        attribute toUnit : ISQBaseUnits;
    }

    part def SystemController {
        part lengthConv : LengthConverter;
        part massConv : MassConverter;
        attribute activeCount : Real;
    }

    part controller : SystemController;

    part converter1 : LengthConverter;
    part converter2 : MassConverter;

    connect controller.lengthConv to converter1;
    connect controller.massConv to converter2;

    attribute converter1.fromUnit : USCustomaryLengthUnits = USCustomaryLengthUnits::inch;
    attribute converter1.toUnit : ISQBaseUnits = ISQBaseUnits::meter;
    attribute converter1.myUnitConverter.factor : Real = 0.0254;

    attribute converter2.fromUnit : USCustomaryMassUnits = USCustomaryMassUnits::pound;
    attribute converter2.toUnit : ISQBaseUnits = ISQBaseUnits::kilogram;
    attribute converter2.myUnitConverter.factor : Real = 0.453592;
}