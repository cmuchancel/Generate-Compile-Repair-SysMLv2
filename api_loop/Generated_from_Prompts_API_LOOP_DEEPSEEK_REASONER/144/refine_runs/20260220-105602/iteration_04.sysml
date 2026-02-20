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
        attribute members : Person[*] {unique};
    }

    part def ManagementSystem {
        attribute sparePartsArchive : SparePart[*] {ordered=false};
        attribute integerStorage : Integer[*] {ordered};
        attribute stringManagement : String[*] {unique};
        attribute personnelRoster : Person[*] {ordered, unique};
        attribute nestedPersonnelSets : PersonnelSet[*] {ordered};
        attribute realNumberArray : Real[4];
    }
}