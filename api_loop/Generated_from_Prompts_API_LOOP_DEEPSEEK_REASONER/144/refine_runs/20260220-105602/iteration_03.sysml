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
        attribute integerStorage : Integer[*] ordered;
        attribute stringManagement : String[*] unique;
        attribute personnelRoster : Person[*] ordered, unique;
        attribute nestedPersonnelSets : Person[*] unique [*] ordered;
        attribute realNumberArray : Real[4];
    }
}