package SparePartsManagement {
    private import ScalarValues::*;

    part def SparePart {
        attribute partId : String;
        attribute partName : String;
        attribute quantity : Integer;
    }

    part def Person {
        attribute personId : String;
        attribute personName : String;
        attribute role : String;
    }

    part def SparePartBag {
        attribute bagSize : Integer;
        part element1 : SparePart;
        part element2 : SparePart;
        part element3 : SparePart;
        part element4 : SparePart;
        part element5 : SparePart;
    }

    part def IntegerList {
        attribute listSize : Integer;
        attribute item1 : Integer;
        attribute item2 : Integer;
        attribute item3 : Integer;
        attribute item4 : Integer;
        attribute item5 : Integer;
    }

    part def StringSet {
        attribute setSize : Integer;
        attribute value1 : String;
        attribute value2 : String;
        attribute value3 : String;
        attribute value4 : String;
    }

    part def PersonOrderedSet {
        attribute setSize : Integer;
        part person1 : Person;
        part person2 : Person;
        part person3 : Person;
        part person4 : Person;
    }

    part def PersonSetEntry {
        attribute entryLabel : String;
        part member1 : Person;
        part member2 : Person;
        part member3 : Person;
    }

    part def ListOfPersonSets {
        attribute listSize : Integer;
        part set1 : PersonSetEntry;
        part set2 : PersonSetEntry;
        part set3 : PersonSetEntry;
    }

    part def RealArray {
        attribute length : Integer = 4;
        attribute value1 : Real;
        attribute value2 : Real;
        attribute value3 : Real;
        attribute value4 : Real;
    }

    part def CollectionManager {
        attribute managerId : String;
        attribute managerName : String;
        part sparePartsBag : SparePartBag;
        part integerList : IntegerList;
        part stringSet : StringSet;
        part personnelSet : PersonOrderedSet;
        part nestedPersonSets : ListOfPersonSets;
        part realArray : RealArray;
    }

    part def BusinessSystem {
        attribute systemId : String;
        attribute systemName : String;
        part collectionManager : CollectionManager;
        part archiveSpareParts : SparePartBag;
        part managedPersonnel : PersonOrderedSet;
        part processingArray : RealArray;
    }

    port def CollectionPort {
        attribute portId : String;
    }

    port def BusinessPort {
        attribute portId : String;
    }

    part def IntegrationSystem {
        attribute integrationId : String;
        port collectionAccess : CollectionPort;
        port businessAccess : BusinessPort;
        part mainSystem : BusinessSystem;
        part backupBag : SparePartBag;
        part backupPersonnel : PersonOrderedSet;
    }
}