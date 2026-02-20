package SparePartsManagement {
    private import ScalarValues::*;

    part def SparePart;

    part def Person;

    item def Bag<T>;

    item def List<T>;

    item def Set<T>;

    item def OrderedSet<T>;

    part def CollectionManager {
        attribute spareParts : Bag<SparePart>;
        attribute integers : List<Integer>;
        attribute strings : Set<String>;
        attribute personnel : OrderedSet<Person>;
        attribute nestedPersonSets : List<Set>;
        attribute realNumbers : Real[4];
    }

    part manager : CollectionManager;
}