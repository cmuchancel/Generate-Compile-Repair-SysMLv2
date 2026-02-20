package CollectionsSystem {
   private import ScalarValues::*;

   part def SparePart;
   part def Person;
   part def BusinessFunction;

   part def SparePartBag {
      part elements : SparePart[*];
   }

   part def IntegerList {
      attribute items : Integer[*];
   }

   part def StringSet {
      attribute items : String[*];
   }

   part def PersonnelOrderedSet {
      part items : Person[*];
   }

   part def PersonSet {
      part members : Person[*];
   }

   part def PersonSetList {
      part sets : PersonSet[*];
   }

   part def RealArray {
      attribute values : Real[4];
   }

   part def CollectionSystem {
      part sparePartsBag : SparePartBag;
      part integerList : IntegerList;
      part stringSet : StringSet;
      part personnelOrderedSet : PersonnelOrderedSet;
      part personnelSetList : PersonSetList;
      part realArray : RealArray;
      part business : BusinessFunction;
   }
}