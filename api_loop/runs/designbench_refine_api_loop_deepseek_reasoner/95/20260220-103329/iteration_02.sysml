package AutomobileDocumentManagement {

    private import ScalarValues::*;

    part def TorqueValue {
        attribute value : Real;
        attribute unit : String = "Nm";
    }

    part def Automobile {
        attribute VIN : String;
        attribute make : String;
        attribute model : String;
        attribute year : Integer;
        attribute torque : TorqueValue;
    }

    part def Car :> Automobile;

    part def Documentation {
        attribute title : String;
        attribute content : String;
        attribute referencedObject : String[1..*];
    }

    part def System {
        part automobiles : Automobile[0..*];
        part documents : Documentation[0..*];
    }
}