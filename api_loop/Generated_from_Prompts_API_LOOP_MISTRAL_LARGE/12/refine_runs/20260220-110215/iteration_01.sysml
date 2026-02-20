package WaterSupplyNetwork {
    private import ScalarValues::*;

    enum def WaterType {
        enum Hot;
        enum Cold;
    }

    part def SpigotBank {
        port hotOut : Spigot;
        port coldOut : Spigot;
    }

    part def Spigot {
        attribute waterType : WaterType;
    }

    part def Faucet {
        port hotIn : FaucetInlet;
        port coldIn : FaucetInlet;
    }

    part def FaucetInlet {
        attribute waterType : WaterType;
    }

    part def WaterSupplySystem {
        part mainSupply : SpigotBank;

        part faucet1 : Faucet;
        part faucet2 : Faucet;

        connect mainSupply.hotOut to faucet1.hotIn;
        connect mainSupply.hotOut to faucet2.hotIn;
        connect mainSupply.coldOut to faucet1.coldIn;
        connect mainSupply.coldOut to faucet2.coldIn;
    }
}