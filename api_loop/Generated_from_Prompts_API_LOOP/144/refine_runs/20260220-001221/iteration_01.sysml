package CollectionsSystem {
   private import ScalarValues::*;
   part def SparePart;
   part def Person;
   part def BusinessFunction;
   part def CollectionSystem {
      attribute sparePartsBag : Bag<SparePart>;
      attribute integerList : List<Integer>;
      attribute stringSet : Set<String>;
      attribute personnelOrderedSet : OrderedSet<Person>;
      attribute personnelSetList : List<Set<Person>>;
      attribute realArray : Real[4];
      part business : BusinessFunction;
   }
}