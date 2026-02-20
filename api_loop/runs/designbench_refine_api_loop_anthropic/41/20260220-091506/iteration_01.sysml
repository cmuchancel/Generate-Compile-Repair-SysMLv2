package ServerSystem {
    private import ScalarValues::*;

    enum def ServerState {
        enum off;
        enum waiting;
        enum response;
    }

    enum def CommandType {
        enum start;
        enum request;
        enum stop;
    }

    item def RequestMessage {
        attribute command : CommandType;
    }

    item def ResponseMessage {
        attribute content : String;
    }

    port def RequestPort {
        attribute receivedCommand : CommandType;
    }

    port def ClockPort {
        attribute hours : Integer;
        attribute minutes : Integer;
        attribute seconds : Integer;
    }

    part def LocalClock {
        attribute currentHours : Integer;
        attribute currentMinutes : Integer;
        attribute currentSeconds : Integer;
        attribute scheduledOffHours : Integer = 23;
        attribute scheduledOffMinutes : Integer = 59;
        attribute scheduledOffSeconds : Integer = 0;
        port clockOut : ClockPort;
    }

    part def Server {
        attribute currentState : ServerState;
        attribute responseWaitMinutes : Integer = 5;
        attribute startupComplete : Boolean;
        attribute dailyScheduleEnabled : Boolean = true;
        attribute cycleCount : Integer;
        attribute maxDailyCycles : Integer;
        port requestPort : RequestPort;
        port serverClockPort : ClockPort;
        part localClock : LocalClock;
    }

    part def RequestSource {
        attribute sourceId : String;
        port outPort : RequestPort;
    }

    part def ServerSystem {
        part server : Server;
        part requestSource : RequestSource;
        connect requestSource.outPort to server.requestPort;
        connect server.localClock.clockOut to server.serverClockPort;
    }
}