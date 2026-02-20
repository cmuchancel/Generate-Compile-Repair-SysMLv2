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

    part def ManagementSystem {
        attribute sparePartsArchive : SparePart[*];
        attribute integerStorage : Integer[*];
        attribute stringManagement : String[*];
        attribute personnelRoster : Person[*];
        attribute nestedPersonnelSets : Person[0..*][*];
        attribute realNumberArray : Real[4];
    }
}