package TrailerHitchingSystem {
  private import ScalarValues::*;
  enum def ConnectionState {
    enum Connected;
    enum Disconnected;
  }
  port def PowerPort {
  }
  part def Frame {
  }
  part def HitchBall {
    port powerPort : PowerPort;
  }
  part def TrailerCoupler {
    port powerPort : PowerPort;
  }
  part def Vehicle {
    part vehicleFrame : Frame;
    part hitchBall : HitchBall;
  }
  part def Trailer {
    part trailerFrame : Frame;
    part trailerCoupler : TrailerCoupler;
  }
  part def HitchingSystem {
    part vehicle : Vehicle;
    part trailer : Trailer;
    attribute connectionState : ConnectionState = ConnectionState::Disconnected;
    connect vehicle.hitchBall.powerPort to trailer.trailerCoupler.powerPort;
  }
  part system : HitchingSystem;
}