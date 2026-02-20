package VehicleLifecycleManagement {
    private import ScalarValues::*;

    enum def LifecycleStage {
        enum Assembly;
        enum Delivery;
        enum Ownership;
        enum EndOfUse;
    }

    part def Vehicle {
        attribute vin : String;
        attribute assemblyDate : Date;
        attribute deliveryDate : Date;
        attribute scrappingDate : Date;
        attribute currentStage : LifecycleStage = LifecycleStage::Assembly;
    }

    part def Owner {
        attribute ownerId : String;
        attribute name : String;
        attribute contact : String;
    }

    part def Driver {
        attribute driverId : String;
        attribute name : String;
        attribute licenseNumber : String;
    }

    part def OwnershipRecord {
        attribute startDate : Date;
        attribute endDate : Date;
        part owner : Owner;
    }

    part def DrivingRecord {
        attribute startTime : DateTime;
        attribute endTime : DateTime;
        part driver : Driver;
    }

    part def VehicleLifecycle {
        part vehicle : Vehicle;
        part ownershipRecords : OwnershipRecord[*];
        part drivingRecords : DrivingRecord[*];
    }

    part lifecycle : VehicleLifecycle;

    connect lifecycle.vehicle to lifecycle.ownershipRecords.vehicle;
    connect lifecycle.vehicle to lifecycle.drivingRecords.vehicle;
}