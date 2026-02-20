package AutomotiveSystem {
    private import ScalarValues::*;

    enum def SafetyStatus {
        enum mandatory;
        enum optional;
    }

    part def AutomotiveSystem {
        part root : Vehicle;
    }

    part def Vehicle {
        part chassis : Chassis;
        part powertrain : Powertrain;
        part safetySystem : SafetySystem;
        part infotainment : Infotainment;
        part body : Body;
    }

    part def Chassis {
        part suspension : Suspension;
        part steering : Steering;
        part braking : Braking;
    }

    part def Powertrain {
        part engine : Engine;
        part transmission : Transmission;
        part drivetrain : Drivetrain;
    }

    part def SafetySystem {
        attribute safetyStatus : SafetyStatus;

        part seatBelts : SeatBelts {
            attribute safetyStatus : SafetyStatus = SafetyStatus::mandatory;
        }
        part bumpers : Bumpers {
            attribute safetyStatus : SafetyStatus = SafetyStatus::mandatory;
        }
        part driverAirbags : DriverAirbags {
            attribute safetyStatus : SafetyStatus = SafetyStatus::optional;
        }
        part abs : ABS {
            attribute safetyStatus : SafetyStatus = SafetyStatus::optional;
        }
    }

    part def Infotainment {
        part headUnit : HeadUnit;
        part speakers : Speakers;
    }

    part def Body {
        part doors : Doors;
        part windows : Windows;
    }

    part def SeatBelts;
    part def Bumpers;
    part def DriverAirbags;
    part def ABS;
    part def Suspension;
    part def Steering;
    part def Braking;
    part def Engine;
    part def Transmission;
    part def Drivetrain;
    part def HeadUnit;
    part def Speakers;
    part def Doors;
    part def Windows;
}