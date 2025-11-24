package 'VehicleTrailerHitching' {
	part def HitchBall;
	part def TrailerCoupler;
part def VehicleFrame {
	part hitchBall : HitchBall;
}
part def TrailerFrame {
	part coupler : TrailerCoupler;
}
part def Vehicle {
	part chassisFrame : VehicleFrame;
}
part def Trailer {
	part towFrame : TrailerFrame;
}
	connection def HitchingConnection {
		end ball : HitchBall[1];
		end coupler : TrailerCoupler[1];
	}
	action def ConnectTrailer {
		in ball : HitchBall;
		in coupler : TrailerCoupler;
	}
	action def DisconnectTrailer {
		in ball : HitchBall;
		in coupler : TrailerCoupler;
	}
	part hitchingSystem {
		part vehicle : Vehicle;
		part trailer : Trailer;
		connection hitchLink : HitchingConnection {
			end ball = vehicle.chassisFrame.hitchBall;
			end coupler = trailer.towFrame.coupler;
		}
		action hitchConnect : ConnectTrailer {
			in ball = vehicle.chassisFrame.hitchBall;
			in coupler = trailer.towFrame.coupler;
		}
		action hitchDisconnect : DisconnectTrailer {
			in ball = vehicle.chassisFrame.hitchBall;
			in coupler = trailer.towFrame.coupler;
		}
	}
}
