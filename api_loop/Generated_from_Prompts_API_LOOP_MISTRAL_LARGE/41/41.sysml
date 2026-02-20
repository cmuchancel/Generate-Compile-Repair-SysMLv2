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

    port def RequestPort {
        attribute request : Request;
    }

    item def Request;
    item def StartCommand;

    part def Server {
        attribute serverState : ServerState = ServerState::off;
        attribute clock : Clock;
        port requestPort : RequestPort;

        attribute lastResponseTime : String;
        attribute dailyResetTime : String = "11:59:00";
    }

    part server : Server;
    part clock : Clock;
    part requestPort : RequestPort;

    connect server.clock to clock;
    connect server.requestPort to requestPort;
}