package ScientificConstants {
  private import ScalarValues::*;

  part def ConstantSet {
    attribute pi : Real = 3.14159265358979323846;
    attribute e : Real = 2.71828182845904523536;
    attribute speedOfLight_m_per_s : Real = 299792458.0;
    attribute fineStructureConstant : Real = 0.00729735256930000000;
    attribute electronToProtonMassRatio : Real = 0.00054461702148600000;
    attribute standardGravity_m_per_s2 : Real = 9.80665;
    attribute amplifierGain_ModelX : Real = 12.5;
  }

  part systemConstants : ConstantSet;
}