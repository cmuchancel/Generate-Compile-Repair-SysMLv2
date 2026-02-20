package VehicleLifecyclePkg {
  private import ScalarValues::*;

  part def VehicleLifecycleSystem {
    part vehicle : Vehicle;
  }

  part def Vehicle {
    attribute vin : String;
    part assembly : AssemblyStage;
    part delivery : DeliveryStage;
    part ownerships : OwnershipRecord[0..*];
    part drivingOperations : DrivingOperation[0..*];
    part scrapping : ScrappingInfo;
  }

  part def AssemblyStage {
    attribute assemblyDate : String;
    attribute assemblyPlant : String;
  }

  part def DeliveryStage {
    attribute deliveryDate : String;
    attribute deliveryLocation : String;
    part deliveredToOwner : Owner;
  }

  part def OwnershipRecord {
    attribute ownershipId : String;
    part owner : Owner;
    attribute startDate : String;
    attribute endDate : String;
  }

  part def Owner {
    attribute ownerId : String;
    attribute ownerName : String;
    attribute ownerContact : String;
  }

  part def Driver {
    attribute driverId : String;
    attribute driverName : String;
    attribute licenseNumber : String;
  }

  part def DrivingOperation {
    attribute operationId : String;
    attribute operationDate : String;
    attribute distance : Real;
    part driver : Driver;
  }

  part def ScrappingInfo {
    attribute scrappingDate : String;
    attribute deregistrationId : String;
    attribute scrappingLocation : String;
  }
}