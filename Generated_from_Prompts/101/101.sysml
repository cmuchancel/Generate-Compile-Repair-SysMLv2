package 'VehicleTrailerCoupling' {
	part def CouplingBall;
	part def HitchConnector;
	part def Hitch {
		part ball : CouplingBall;
		part connector : HitchConnector;
	}
	part def Vehicle {
		ref part couplingBall : CouplingBall;
	}
	part def Trailer {
		ref part hitchConnector : HitchConnector;
	}
	connection def CouplingInterface {
		end vehicle : Vehicle[1];
		end trailer : Trailer[1];
		end hitch : Hitch[1];
	}
	part couplingAssembly {
		part vehicle : Vehicle;
		part trailer : Trailer;
		part hitch : Hitch;
		connection couplingLink : CouplingInterface {
			end vehicle = couplingAssembly::vehicle;
			end trailer = couplingAssembly::trailer;
			end hitch = couplingAssembly::hitch;
		}
		bind vehicle.couplingBall = hitch.ball;
		bind trailer.hitchConnector = hitch.connector;
	}
}
