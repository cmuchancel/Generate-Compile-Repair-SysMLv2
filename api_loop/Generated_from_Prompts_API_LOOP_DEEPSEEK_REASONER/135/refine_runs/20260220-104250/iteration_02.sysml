package ScientificConstants {
    private import ScalarValues::*;

    part def Constants {
        attribute e : Real = 2.71828182845904523536;
        attribute pi : Real = 3.14159265358979323846;
        attribute speedOfLightInVacuum : Real = 299792458.0;
        attribute fineStructureConstant : Real = 0.0072973525693;
        attribute electronProtonMassRatio : Real = 0.000544617021487;
        attribute standardGravitationalAcceleration : Real = 9.80665;
        attribute amplifierGainModelX : Real = 100.0;
    }

    part constants : Constants;
}