package VehicleFeatureAnnotationSystem {
    private import ScalarValues::*;

    part def Vehicle {
        part interior : Interior;
        part bodyAssembly : BodyAssembly;
    }

    part def Interior {
        part seatBelts : SeatBelts;
        part frontSeats : FrontSeats;
        part driverAirbag : DriverAirbag;
        part alarmSystem : AlarmSystem;
    }

    part def BodyAssembly {
        part bodyShell : BodyShell;
        part bumpers : Bumpers;
        part keylessEntry : KeylessEntryDevice;
    }

    part def SeatBelts {
        attribute safetyFunction : String = "occupant restraint";
        attribute securityFunction : String = "none";
    }

    part def FrontSeats {
        attribute safetyFunction : String = "occupant positioning";
        attribute securityFunction : String = "none";
    }

    part def DriverAirbag {
        attribute safetyFunction : String = "impact protection";
        attribute securityFunction : String = "none";
    }

    part def AlarmSystem {
        attribute safetyFunction : String = "none";
        attribute securityFunction : String = "theft deterrence";
    }

    part def BodyShell {
        attribute safetyFunction : String = "structural integrity";
        attribute securityFunction : String = "physical barrier";
    }

    part def Bumpers {
        attribute safetyFunction : String = "impact absorption";
        attribute securityFunction : String = "none";
    }

    part def KeylessEntryDevice {
        attribute safetyFunction : String = "none";
        attribute securityFunction : String = "access control";
    }
}