package ScientificConstants {
    private import ScalarValues::*;

    part def MathematicalConstants {
        attribute pi : Real = 3.14159265358979323846;
        attribute e : Real = 2.71828182845904523536;
    }

    part def FundamentalPhysicalConstants {
        attribute speedOfLight : Real = 299792458.0; // m/s
        attribute fineStructureConstant : Real = 0.0072973525693;
        attribute electronProtonMassRatio : Real = 0.000544617021487;
    }

    part def ContextSpecificConstants {
        attribute standardGravity : Real = 9.80665; // m/s^2
        attribute amplifierGainModelX : Real = 100.0; // unitless
    }

    part def ScientificConstantsSystem {
        part mathConstants : MathematicalConstants;
        part physicalConstants : FundamentalPhysicalConstants;
        part scenarioConstants : ContextSpecificConstants;
    }
}