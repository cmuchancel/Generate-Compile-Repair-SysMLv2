package TimedClockBehavior {
    private import ScalarValues::*;

    enum def StateKind {
        enum off;
        enum waiting;
        enum response;
    }

    enum def MessageKind {
        enum start;
        enum request;
    }

    item def Message {
        attribute kind : MessageKind;
    }

    port def RequestPort {
        attribute received : Message[0..*];
    }

    part def Clock {
        attribute currentTime : Time;
        attribute lastMidnight : Time;
    }

    part def Server {
        attribute currentState : StateKind = StateKind::off;
        attribute responseStart : Time;
        port requestPort : RequestPort;
        part mainClock : Clock;
    }

    part server : Server;
}