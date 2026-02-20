package CollectionsManagement {
    private import Base::Collections::*;
    private import ScalarValues::*;

    item def Person;
    item def SparePart;

    part def CollectionManager {
        attribute spareParts : Bag;
        attribute integerValues : List;
        attribute stringValues : Set;
        attribute personnel : OrderedSet;
        attribute nestedPersonnelSets : List;
        attribute realValues : Real[4];
    }

    part system : CollectionManager;
}