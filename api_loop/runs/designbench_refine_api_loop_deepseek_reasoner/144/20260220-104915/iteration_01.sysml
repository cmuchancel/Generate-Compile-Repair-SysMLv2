package CollectionManagement {
    private import ScalarValues::*;
    
    part def Person {
        attribute name : String;
        attribute id : Integer;
    }
    
    part def SparePart {
        attribute partNumber : String;
        attribute description : String;
        attribute quantity : Integer;
    }
    
    part def System {
        attribute sparePartsBag : Bag<SparePart>;
        attribute integerList : List<Integer>;
        attribute stringSet : Set<String>;
        attribute personnelSet : OrderedSet<Person>;
        attribute nestedPersonnel : List<Set<Person>>;
        attribute realArray : Real[4];
    }
}