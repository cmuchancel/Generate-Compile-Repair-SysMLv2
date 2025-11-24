package 'WaterSupplyNetwork' {
	item def HotWater;
	item def ColdWater;
	
	port def Spigot {
		out item water;
	}
	
	port def FaucetInlet {
		in item water;
	}
	
	port def HotSpigot :> Spigot {
		out item redefines water : HotWater;
	}
	
	port def ColdSpigot :> Spigot {
		out item redefines water : ColdWater;
	}
	
	port def HotFaucetInlet :> FaucetInlet {
		in item redefines water : HotWater;
	}
	
	port def ColdFaucetInlet :> FaucetInlet {
		in item redefines water : ColdWater;
	}
	
	part def SpigotBank {
		port hotOutlet : HotSpigot;
		port coldOutlet : ColdSpigot;
	}
	
	part def Faucet {
		port hotInlet : HotFaucetInlet;
		port coldInlet : ColdFaucetInlet;
	}
	
	part waterDistribution {
		part spigotBank : SpigotBank;
		part faucet : Faucet[2];
		connect spigotBank.hotOutlet to faucet.hotInlet;
		connect spigotBank.coldOutlet to faucet.coldInlet;
	}
}
