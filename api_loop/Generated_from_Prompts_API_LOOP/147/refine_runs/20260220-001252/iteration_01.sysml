package AlloySystem {
  private import ScalarValues::*;

  part def Material {
    attribute name : String;
    attribute atomicMass : Real;
  }

  part def Component {
    part material : Material;
    attribute massFraction : Real;
  }

  part def Alloy {
    part components : Component[2..*];
    attribute tensileStrength : Real;
  }

  part iron : Material {
    attribute name = "Iron";
    attribute atomicMass = 55.845;
  }

  part carbon : Material {
    attribute name = "Carbon";
    attribute atomicMass = 12.011;
  }

  part manganese : Material {
    attribute name = "Manganese";
    attribute atomicMass = 54.938;
  }

  part steel980 : Alloy {
    attribute tensileStrength = 980;
    part compIron : Component {
      part material = iron;
      attribute massFraction = 0.98;
    }
    part compCarbon : Component {
      part material = carbon;
      attribute massFraction = 0.02;
    }
  }
}