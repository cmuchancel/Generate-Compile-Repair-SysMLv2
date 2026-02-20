package AutomobileDocumentManagement {
  private import ScalarValues::*;

  item def Document {
    attribute title : String;
    attribute content : String;
  }

  part def Automobile {
    attribute vin : String;
    attribute model : String;
    attribute torque : TorqueValue;
    attribute description : String;
  }

  part def Car {
    attribute aliasName : String = "Car";
    attribute automobile : Automobile;
    attribute description : String;
  }

  part def DocumentationLink {
    attribute document : Document;
    attribute targetAutomobile : Automobile;
    attribute targetCar : Car;
  }

  part def Platform {
    part automobiles : Automobile[0..*];
    part cars : Car[0..*];
    part documents : Document[0..*];
    part links : DocumentationLink[0..*];
  }
}