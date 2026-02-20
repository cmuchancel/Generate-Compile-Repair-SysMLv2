package TimedServerSystem {
    private import ScalarValues::*;

    enum def ServerState {
        enum off;
        enum waiting;
        enum response;
    }

    part def Clock {
        attribute currentTime : String;
    }

    part def RequestPort {
        port in request : Request;
    }

    item def Request;
    item def StartCommand;

    part def Server {
        attribute state : ServerState = ServerState::off;
        attribute localClock : Clock;
        port requestPort : RequestPort;

        attribute lastResponseTime : String;
        attribute dailyResetTime : String = "11:59:00";
    }

    part server : Server;
    part clock : Clock;
    part requestPort : RequestPort;

    connect server.localClock to clock;
    connect server.requestPort to requestPort;
}