package CollectionManager {
    private import ScalarValues::*;
    private import Kernel::Collections::*;

    part def SparePart;

    item def Person;

    part def System {
        attribute spareParts : Bag<SparePart>;
        attribute integerList : List<Integer>;
        attribute stringSet : Set<String>;
        attribute personnel : OrderedSet<Person>;
        attribute nestedPersonnelList : List<Set<Person>>;
        attribute realArray : Sequence<Real, 4>;
    }
}