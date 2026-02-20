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
        attribute sparePartsBag : SparePart[0..*] { nonunique, unordered; }
        attribute integerList : Integer[0..*] { ordered; }
        attribute stringSet : String[0..*];
        attribute personnelSet : Person[0..*] { ordered; }
        attribute nestedPersonnel : Person[0..*][0..*];
        attribute realArray : Real[4];
    }
}