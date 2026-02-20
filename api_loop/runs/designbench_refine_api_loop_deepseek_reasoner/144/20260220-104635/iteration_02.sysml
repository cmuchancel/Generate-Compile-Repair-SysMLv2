package CollectionsManagement {
    private import ScalarValues::*;

    item def Person;

    item def SparePart;

    part def CollectionManager {
        attribute spareParts : Bag<SparePart>;
        attribute integerValues : List<Integer>;
        attribute stringValues : Set<String>;
        attribute personnel : OrderedSet<Person>;
        attribute nestedPersonnelSets : List<Set<Person>>;
        attribute realValues : Real[4];
    }

    part system : CollectionManager;
}