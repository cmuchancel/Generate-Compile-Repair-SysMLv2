package TimedServer {
  private import ScalarValues::*;

  enum def ServerState {
    enum off;
    enum waiting;
    enum response;
  }

  enum def CommandType {
    enum start;
    enum request;
  }

  item def RequestMessage {
    attribute cmd : CommandType;
  }

  part def LocalClock {
    attribute currentTimeSeconds : Real;
    attribute daySeconds : Real = 86400.0;
  }

  port def RequestPort {
    attribute lastMessage : RequestMessage;
  }

  part def Server {
    attribute serverState : ServerState = ServerState::off;
    attribute responseDelaySeconds : Real = 300.0;
    attribute dailyOffTimeSeconds : Real = 43140.0;
    part clock : LocalClock;
    port requestPort : RequestPort;
  }

  part def System {
    part server : Server;
  }
}