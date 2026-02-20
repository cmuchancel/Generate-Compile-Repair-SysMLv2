package AutomobileDocumentManagement {
    
    private import ScalarValues::*;
    
    part def Automobile {
        attribute VIN : string;
        attribute make : string;
        attribute model : string;
        attribute year : integer;
        attribute torque : TorqueValue;
    }
    
    part def Car {
        alias Automobile;
    }
    
    part def Documentation {
        attribute title : string;
        attribute content : string;
        attribute referencedObject : string[1..*];
    }
    
    part def System {
        part automobiles : Automobile[0..*];
        part cars : Car[0..*];
        part documents : Documentation[0..*];
    }
}