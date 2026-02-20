package CollectionManagement {
    private import ScalarValues::*;

    item def SparePart {
        attribute id : String;
        attribute description : String;
    }

    item def Person {
        attribute name : String;
        attribute employeeId : Integer;
    }

    item def PersonnelSet {
        attribute members : Person[*];
    }

    part def ManagementSystem {
        attribute sparePartsArchive : SparePart[*];
        attribute integerStorage : Integer[*];
        attribute stringManagement : String[*];
        attribute personnelRoster : Person[*];
        attribute nestedPersonnelSets : PersonnelSet[*];
        attribute realNumberArray : Real[4];
    }
}