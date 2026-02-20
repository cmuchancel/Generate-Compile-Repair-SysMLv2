package TimedClockBehavior {
    private import ScalarValues::*;

    enum def StateKind {
        off,
        waiting,
        response
    }

    item def Message {
        attribute kind : String;
    }

    port def RequestPort {
        attribute received : Message[0..*];
    }

    part def Clock {
        attribute currentTime : Time;
        attribute lastMidnight : Time;
    }

    part def Server {
        attribute state : StateKind = StateKind::off;
        attribute responseStart : Time;
        port requestPort : RequestPort;
        part localClock : Clock;
    }

    part server : Server;
}