package AutomobileDocumentManagement {
    private import ScalarValues::*;

    part def Automobile {
        attribute documentId : String;
        attribute make : String;
        attribute model : String;
        attribute year : String;
        attribute description : String;
    }

    part def Car {
        attribute documentId : String;
        attribute make : String;
        attribute model : String;
        attribute year : String;
        attribute description : String;
    }

    enum def DocumentStatus {
        enum Active;
        enum Archived;
        enum Pending;
    }

    part def TorqueValue {
        attribute magnitude : Real;
        attribute unit : String;
        attribute isqStandard : String;
    }

    part def DocumentRecord {
        attribute recordId : String;
        attribute title : String;
        attribute content : String;
        attribute status : DocumentStatus;
        attribute annotation : String;
    }

    part def PhysicalQuantityReference {
        attribute quantityName : String;
        attribute isqStandard : String;
        attribute quantityDescription : String;
    }

    part def AutomobileDocument {
        part vehicle : Automobile;
        part torqueSpec : TorqueValue;
        part record : DocumentRecord;
        part quantityRef : PhysicalQuantityReference;
    }

    part def CarDocument {
        part vehicle : Car;
        part torqueSpec : TorqueValue;
        part record : DocumentRecord;
        part quantityRef : PhysicalQuantityReference;
    }

    part def DocumentManagementPlatform {
        attribute platformName : String;
        attribute platformDescription : String;
        part automobileDoc : AutomobileDocument;
        part carDoc : CarDocument;
    }

    part platform : DocumentManagementPlatform {
        part automobileDoc : AutomobileDocument {
            part vehicle : Automobile {
                attribute documentId : String = "AUTO-001";
                attribute make : String = "Generic";
                attribute model : String = "Model";
                attribute year : String = "2024";
                attribute description : String = "Standard automobile record";
            }
            part torqueSpec : TorqueValue {
                attribute magnitude : Real = 250.0;
                attribute unit : String = "Nm";
                attribute isqStandard : String = "ISQ";
            }
            part record : DocumentRecord {
                attribute recordId : String = "REC-001";
                attribute title : String = "Automobile Technical Document";
                attribute content : String = "Technical specifications and documentation";
                attribute status : DocumentStatus = DocumentStatus::Active;
                attribute annotation : String = "ISQ compliant torque values referenced";
            }
            part quantityRef : PhysicalQuantityReference {
                attribute quantityName : String = "TorqueValue";
                attribute isqStandard : String = "ISQ";
                attribute quantityDescription : String = "Torque quantity as defined under ISQ standard";
            }
        }

        part carDoc : CarDocument {
            part vehicle : Car {
                attribute documentId : String = "CAR-001";
                attribute make : String = "Generic";
                attribute model : String = "Model";
                attribute year : String = "2024";
                attribute description : String = "Car alias record for automobile";
            }
            part torqueSpec : TorqueValue {
                attribute magnitude : Real = 250.0;
                attribute unit : String = "Nm";
                attribute isqStandard : String = "ISQ";
            }
            part record : DocumentRecord {
                attribute recordId : String = "REC-002";
                attribute title : String = "Car Technical Document";
                attribute content : String = "Technical specifications using Car alias";
                attribute status : DocumentStatus = DocumentStatus::Active;
                attribute annotation : String = "Car is an alias for Automobile in this platform";
            }
            part quantityRef : PhysicalQuantityReference {
                attribute quantityName : String = "TorqueValue";
                attribute isqStandard : String = "ISQ";
                attribute quantityDescription : String = "Torque quantity as defined under ISQ standard";
            }
        }
    }
}