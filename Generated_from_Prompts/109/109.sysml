package 'VehicleTrailerDynamicReference' {
	private import ScalarValues::*;
	part def TowBall;
	part def VehicleChassis {
		part towBall : TowBall;
	}
	part def Vehicle {
		part chassis : VehicleChassis;
	}
	part def TrailerChassis;
part def CouplerMechanism {
	attribute isLatched : Boolean;
}
	part def Trailer {
		part chassis : TrailerChassis;
		part coupler : CouplerMechanism;
	}
	connection def HitchReference {
		end towBall : TowBall[1];
		end coupler : CouplerMechanism[1];
	}
action def HitchTrailer {
	in towBall : TowBall;
	in coupler : CouplerMechanism;
}
action def UnhitchTrailer {
	in coupler : CouplerMechanism;
}
	part hitchReferenceModel {
		part vehicle : Vehicle;
		part trailer : Trailer;
		connection hitchLink : HitchReference {
			end towBall = vehicle.chassis.towBall;
			end coupler = trailer.coupler;
		}
		action hitch : HitchTrailer {
			in towBall = vehicle.chassis.towBall;
			in coupler = trailer.coupler;
		}
		action unhitch : UnhitchTrailer {
			in coupler = trailer.coupler;
		}
	}
}
