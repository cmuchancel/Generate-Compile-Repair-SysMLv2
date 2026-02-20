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
        attribute safetyFunction : string = "occupant restraint";
        attribute securityFunction : string = "none";
    }

    part def FrontSeats {
        attribute safetyFunction : string = "occupant positioning";
        attribute securityFunction : string = "none";
    }

    part def DriverAirbag {
        attribute safetyFunction : string = "impact protection";
        attribute securityFunction : string = "none";
    }

    part def AlarmSystem {
        attribute safetyFunction : string = "none";
        attribute securityFunction : string = "theft deterrence";
    }

    part def BodyShell {
        attribute safetyFunction : string = "structural integrity";
        attribute securityFunction : string = "physical barrier";
    }

    part def Bumpers {
        attribute safetyFunction : string = "impact absorption";
        attribute securityFunction : string = "none";
    }

    part def KeylessEntryDevice {
        attribute safetyFunction : string = "none";
        attribute securityFunction : string = "access control";
    }
}