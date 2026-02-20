package MaterialsDB {
  private import ScalarValues::*;

  enum def MaterialType { metal; alloy; }

  part def Material {
    attribute name : String;
    attribute materialType : MaterialType;
    attribute atomicMass : Real;
    attribute tensileStrength_N_per_mm2 : Real;
  }

  part def ConstituentFraction {
    attribute constituentName : String;
    attribute massFraction : Real;
  }

  part def MetalAlloy : Material {
    part constituents[*] : ConstituentFraction;
    attribute compositionalInfo : String;
  }

  part Steel_980 : MetalAlloy {
    attribute name = "Steel_980";
    attribute materialType = MaterialType::alloy;
    attribute atomicMass = 55.9;
    attribute tensileStrength_N_per_mm2 = 980.0;
    attribute compositionalInfo = "High-strength steel alloy";
    part Fe : ConstituentFraction {
      attribute constituentName = "Iron";
      attribute massFraction = 0.985;
    }
    part C : ConstituentFraction {
      attribute constituentName = "Carbon";
      attribute massFraction = 0.010;
    }
    part Mn : ConstituentFraction {
      attribute constituentName = "Manganese";
      attribute massFraction = 0.005;
    }
  }
}