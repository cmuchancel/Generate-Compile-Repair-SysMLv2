package VehicleLifecycleManagement {
    private import ScalarValues::*;

    enum def LifecycleStage {
        enum Assembly;
        enum Delivery;
        enum InUse;
        enum Scrapped;
    }

    enum def OwnershipStatus {
        enum Active;
        enum Transferred;
        enum Terminated;
    }

    part def Vehicle {
        attribute vehicleId : String;
        attribute make : String;
        attribute model : String;
        attribute year : Integer;
        attribute currentStage : LifecycleStage;
    }

    part def AssemblyRecord {
        attribute assemblyId : String;
        attribute assemblyDate : String;
        attribute assemblyLocation : String;
        attribute assemblyNotes : String;
    }

    part def DeliveryRecord {
        attribute deliveryId : String;
        attribute deliveryDate : String;
        attribute deliveryLocation : String;
        attribute recipientName : String;
    }

    part def Owner {
        attribute ownerId : String;
        attribute ownerName : String;
        attribute ownerContact : String;
        attribute ownerAddress : String;
    }

    part def OwnershipRecord {
        attribute ownershipId : String;
        attribute ownershipStartDate : String;
        attribute ownershipEndDate : String;
        attribute ownershipStatus : OwnershipStatus;
        part owner : Owner;
    }

    part def Driver {
        attribute driverId : String;
        attribute driverName : String;
        attribute licenseNumber : String;
        attribute driverContact : String;
    }

    part def DrivingRecord {
        attribute drivingRecordId : String;
        attribute drivingDate : String;
        attribute startLocation : String;
        attribute endLocation : String;
        attribute distanceDriven : Real;
        part actualDriver : Driver;
    }

    part def ScrapRecord {
        attribute scrapId : String;
        attribute scrapDate : String;
        attribute scrapReason : String;
        attribute scrapLocation : String;
        attribute deregistrationNumber : String;
    }

    part def OwnershipPeriod {
        attribute periodId : String;
        part ownershipRecord : OwnershipRecord;
        part drivingRecords : DrivingRecord[0..*];
    }

    part def VehicleLifecycle {
        attribute lifecycleId : String;
        part vehicle : Vehicle;
        part assemblyRecord : AssemblyRecord;
        part deliveryRecord : DeliveryRecord;
        part ownershipPeriods : OwnershipPeriod[0..*];
        part scrapRecord : ScrapRecord[0..1];
    }

    port def VehicleInfoPort {
        attribute vehicleId : String;
        attribute stage : LifecycleStage;
    }

    port def OwnershipPort {
        attribute ownershipId : String;
        attribute ownerId : String;
    }

    port def DrivingPort {
        attribute drivingRecordId : String;
        attribute driverId : String;
    }

    part def LifecycleTracker {
        attribute trackerId : String;
        port vehicleInfo : VehicleInfoPort;
        port ownershipInfo : OwnershipPort;
        port drivingInfo : DrivingPort;
        part managedLifecycles : VehicleLifecycle[1..*];
    }

    part def LifecycleManagementSystem {
        attribute systemId : String;
        attribute systemName : String;
        part tracker : LifecycleTracker;
    }

    part lifecycleSystem : LifecycleManagementSystem {
        attribute systemId = "SYS-001";
        attribute systemName = "Vehicle Lifecycle Management System";
        part tracker : LifecycleTracker {
            attribute trackerId = "TRACKER-001";
            part managedLifecycles : VehicleLifecycle[1..*];
        }
    }
}