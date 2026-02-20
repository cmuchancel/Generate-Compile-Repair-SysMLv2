package SparePartsManagement {
    private import ScalarValues::*;

    part def SparePart;

    part def Person;

    item def Bag {
        attribute elements : SparePart[*];
    }

    item def IntList {
        attribute items : Integer[*];
    }

    item def StringSet {
        attribute members : String[*];
    }

    item def PersonOrderedSet {
        attribute elements : Person[*];
    }

    item def NestedPersonSetList {
        attribute sets : StringSet[*];
    }

    part def CollectionManager {
        attribute spareParts : Bag;
        attribute integers : IntList;
        attribute strings : StringSet;
        attribute personnel : PersonOrderedSet;
        attribute nestedPersonSets : IntList;
        attribute realNumbers : Real[4];
    }

    part manager : CollectionManager;
}