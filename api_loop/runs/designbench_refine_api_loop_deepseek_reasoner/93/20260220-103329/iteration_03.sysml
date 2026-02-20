package AutomotiveSystem {
    private import ScalarValues::*;

    enum def SafetyStatus {
        enum mandatory;
        enum optional;
    }

    port def MechanicalInterface;
    port def ElectricalInterface;
    port def SignalInterface;

    part def AutomotiveSystem {
        part chassis : Chassis;
        part safetySubsystem : SafetySubsystem;
        part powertrain : Powertrain;
        part interior : Interior;
        port mechanical : MechanicalInterface;
        port electrical : ElectricalInterface;
        port signals : SignalInterface;

        connect chassis.mechanical to mechanical;
        connect safetySubsystem.electrical to electrical;
        connect safetySubsystem.signals to signals;
        connect powertrain.mechanical to mechanical;
        connect powertrain.electrical to electrical;
        connect interior.mechanical to mechanical;
        connect interior.electrical to electrical;
        connect interior.signals to signals;
    }

    part def Chassis {
        part framePart : Frame;
        part bumper : Bumper;
        part wheel1 : Wheel;
        part wheel2 : Wheel;
        part wheel3 : Wheel;
        part wheel4 : Wheel;
        part absPart : ABS;
        port mechanical : MechanicalInterface;

        connect framePart.mechanical to mechanical;
        connect bumper.mechanical to mechanical;
        connect wheel1.mechanical to mechanical;
        connect wheel2.mechanical to mechanical;
        connect wheel3.mechanical to mechanical;
        connect wheel4.mechanical to mechanical;
        connect absPart.mechanical to mechanical;
    }

    part def SafetySubsystem {
        part seatBelt1 : SeatBelt;
        part seatBelt2 : SeatBelt;
        part seatBelt3 : SeatBelt;
        part seatBelt4 : SeatBelt;
        part driverAirbag : DriverAirbag;
        part passengerAirbag : PassengerAirbag;
        part absPart : ABS;
        port electrical : ElectricalInterface;
        port signals : SignalInterface;

        connect seatBelt1.electrical to electrical;
        connect seatBelt2.electrical to electrical;
        connect seatBelt3.electrical to electrical;
        connect seatBelt4.electrical to electrical;
        connect driverAirbag.electrical to electrical;
        connect passengerAirbag.electrical to electrical;
        connect absPart.signals to signals;
    }

    part def Powertrain {
        part engine : Engine;
        part transmission : Transmission;
        part brakes : Brakes;
        port mechanical : MechanicalInterface;
        port electrical : ElectricalInterface;

        connect engine.mechanical to mechanical;
        connect engine.electrical to electrical;
        connect transmission.mechanical to mechanical;
        connect brakes.mechanical to mechanical;
    }

    part def Interior {
        part seat1 : Seat;
        part seat2 : Seat;
        part seat3 : Seat;
        part seat4 : Seat;
        part dashboard : Dashboard;
        part steering : Steering;
        port mechanical : MechanicalInterface;
        port electrical : ElectricalInterface;
        port signals : SignalInterface;

        connect seat1.mechanical to mechanical;
        connect seat2.mechanical to mechanical;
        connect seat3.mechanical to mechanical;
        connect seat4.mechanical to mechanical;
        connect dashboard.electrical to electrical;
        connect dashboard.signals to signals;
        connect steering.mechanical to mechanical;
        connect steering.electrical to electrical;
        connect steering.signals to signals;
    }

    part def Frame {
        attribute material : String = "Steel";
        port mechanical : MechanicalInterface;
    }

    part def Bumper {
        attribute safetyStatus : SafetyStatus = mandatory;
        attribute material : String = "Plastic";
        port mechanical : MechanicalInterface;
    }

    part def Wheel {
        attribute size : Integer = 18;
        port mechanical : MechanicalInterface;
    }

    part def SeatBelt {
        attribute safetyStatus : SafetyStatus = mandatory;
        attribute type : String = "Three-point";
        port electrical : ElectricalInterface;
    }

    part def DriverAirbag {
        attribute safetyStatus : SafetyStatus = optional;
        attribute location : String = "Steering Wheel";
        port electrical : ElectricalInterface;
    }

    part def PassengerAirbag {
        attribute safetyStatus : SafetyStatus = optional;
        attribute location : String = "Dashboard";
        port electrical : ElectricalInterface;
    }

    part def ABS {
        attribute safetyStatus : SafetyStatus = optional;
        attribute version : String = "MK60";
        port mechanical : MechanicalInterface;
        port signals : SignalInterface;
    }

    part def Engine {
        attribute displacement : Integer = 2000;
        port mechanical : MechanicalInterface;
        port electrical : ElectricalInterface;
    }

    part def Transmission {
        attribute type : String = "Automatic";
        port mechanical : MechanicalInterface;
    }

    part def Brakes {
        attribute type : String = "Disc";
        port mechanical : MechanicalInterface;
    }

    part def Seat {
        attribute material : String = "Leather";
        port mechanical : MechanicalInterface;
    }

    part def Dashboard {
        attribute displayType : String = "Digital";
        port electrical : ElectricalInterface;
        port signals : SignalInterface;
    }

    part def Steering {
        attribute type : String = "Power Steering";
        port mechanical : MechanicalInterface;
        port electrical : ElectricalInterface;
        port signals : SignalInterface;
    }
}