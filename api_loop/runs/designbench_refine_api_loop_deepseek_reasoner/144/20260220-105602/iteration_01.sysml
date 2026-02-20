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
        attribute sparePartsArchive : Bag<SparePart>;
        attribute integerStorage : List<Integer>;
        attribute stringManagement : Set<String>;
        attribute personnelRoster : OrderedSet<Person>;
        attribute nestedPersonnelSets : List<Set<Person>>;
        attribute realNumberArray : Real[4];
    }
}