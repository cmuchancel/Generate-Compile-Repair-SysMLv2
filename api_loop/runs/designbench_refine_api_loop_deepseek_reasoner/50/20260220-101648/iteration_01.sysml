package VehicleLifecycleManagement {
    private import ScalarValues::*;

    part def Vehicle {
        attribute vin : String;
        attribute assemblyDate : String;
        attribute deliveryDate : String;
        part ownershipHistory : OwnershipRecord[0..*];
        part drivingHistory : DrivingEvent[0..*];
        part scrappingRecord : ScrappingRecord[0..1];
    }

    part def OwnershipRecord {
        attribute ownerId : String;
        attribute ownerName : String;
        attribute ownerAddress : String;
        attribute startDate : String;
        attribute endDate : String;
    }

    part def DrivingEvent {
        attribute driverId : String;
        attribute driverName : String;
        attribute driverLicenseNumber : String;
        attribute date : String;
        attribute startTime : String;
        attribute endTime : String;
    }

    part def ScrappingRecord {
        attribute date : String;
        attribute reason : String;
        attribute facility : String;
    }
}