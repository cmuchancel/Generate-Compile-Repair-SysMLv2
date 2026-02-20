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
        attribute sparePartsArchive : SparePart[*] {nonunique};
        attribute integerStorage : Integer[*] {ordered, nonunique};
        attribute stringManagement : String[*];
        attribute personnelRoster : Person[*] {ordered};
        attribute nestedPersonnelSets : Person[*][*];
        attribute realNumberArray : Real[4];
    }
}