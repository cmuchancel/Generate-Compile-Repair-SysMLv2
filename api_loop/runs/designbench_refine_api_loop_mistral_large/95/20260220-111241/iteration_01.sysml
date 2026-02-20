package AutomobileDocumentManagement {
    private import ScalarValues::*;

    enum def DocumentType {
        enum Specification;
        enum Manual;
        enum ServiceRecord;
        enum ComplianceCertificate;
    }

    part def PhysicalQuantity {
        attribute name : String;
        attribute symbol : String;
        attribute unit : String;
    }

    part def ISQ {
        part torqueValue : PhysicalQuantity;
    }

    part def Automobile {
        attribute vin : String;
        attribute make : String;
        attribute model : String;
        attribute year : Integer;
        attribute documents : DocumentType[0..*];
        part isq : ISQ;
    }

    part def Car :> Automobile;

    part def Document {
        attribute title : String;
        attribute content : String;
        attribute docType : DocumentType;
        attribute associatedAutomobile : Automobile;
    }

    part def DocumentManagementSystem {
        part automobiles : Automobile[0..*];
        part cars : Car[0..*];
        part documents : Document[0..*];

        attribute isq : ISQ;

        port def DocumentInputPort;
        port def DocumentQueryPort;
        port def DocumentOutputPort;

        port input : DocumentInputPort;
        port query : DocumentQueryPort;
        port output : DocumentOutputPort;
    }

    part system : DocumentManagementSystem;

    connect system.input to system.documents;
    connect system.query to system.automobiles;
    connect system.query to system.cars;
    connect system.output to system.documents;
}