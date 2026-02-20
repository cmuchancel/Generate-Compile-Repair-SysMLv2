package 'WaterSupplyNetwork12' {
	private import ScalarValues::*;

	part def WaterSupplyNetwork {
		part supplyPipe : SpigotBank;
		part faucets : Faucet;
		attribute simultaneousSupplySupported : Boolean;
		attribute pipelineConnectionIntegrity : Boolean;
		attribute pipelineTransmissionContinuity : Boolean;
	}

	part def SpigotBank {
		port hotOutlet : Spigot;
		port coldOutlet : Spigot;
	}

	part def Faucet {
		port hotInlet : FaucetInlet;
		port coldInlet : FaucetInlet;
	}

	port def Spigot {
		attribute hotWaterDelivery : FlowRate;
		attribute coldWaterDelivery : FlowRate;
	}

	port def FaucetInlet {
		attribute inletFlow : FlowRate;
	}

	attribute def FlowRate {
		attribute volumePerTime : Real;
	}

	connection def HotWaterRoute {
		end source : SpigotBank::hotOutlet;
		end sink : Faucet::hotInlet;
	}

	connection def ColdWaterRoute {
		end source : SpigotBank::coldOutlet;
		end sink : Faucet::coldInlet;
	}
}
