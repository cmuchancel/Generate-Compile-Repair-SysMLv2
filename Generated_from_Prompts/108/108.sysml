package 'VehicleTrailerConnectionStatus' {
	enum def ConnectionState {
		enum connected;
		enum disconnected;
	}
	part def TowBall;
	part def TrailerCoupler;
	part def VehicleChassis {
		part towBall : TowBall;
	}
	part def TrailerFrame {
		part coupler : TrailerCoupler;
	}
	part def Vehicle {
		part chassis : VehicleChassis;
	}
part def Trailer {
	part towStructure : TrailerFrame;
}
	connection def HitchInterface {
		end towBall : TowBall[1];
		end coupler : TrailerCoupler[1];
	}
	action def HitchTrailer {
		in towBall : TowBall;
		in coupler : TrailerCoupler;
	}
	action def UnhitchTrailer {
		in towBall : TowBall;
		in coupler : TrailerCoupler;
	}
	part connectionManager {
		part vehicle : Vehicle;
		part trailer : Trailer;
		connection hitch : HitchInterface {
			end towBall = vehicle.chassis.towBall;
			end coupler = trailer.towStructure.coupler;
		}
		attribute status : ConnectionState = ConnectionState::disconnected;
		action hitchTrailer : HitchTrailer {
			in towBall = vehicle.chassis.towBall;
			in coupler = trailer.towStructure.coupler;
		}
		action unhitchTrailer : UnhitchTrailer {
			in towBall = vehicle.chassis.towBall;
			in coupler = trailer.towStructure.coupler;
		}
	}
}
