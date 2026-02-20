package SparePartsManagement {
    private import ScalarValues::*;

    part def SparePart;

    part def Person;

    part def CollectionManager {
        attribute spareParts : Bag<SparePart>;
        attribute integers : List<Integer>;
        attribute strings : Set<String>;
        attribute personnel : OrderedSet<Person>;
        attribute nestedPersonSets : List<Set<Person>>;
        attribute realNumbers : Real[4];
    }

    part manager : CollectionManager;
}