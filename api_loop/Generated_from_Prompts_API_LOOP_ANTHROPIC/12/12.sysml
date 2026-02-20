package WaterSupplyNetwork {
    private import ScalarValues::*;

    enum def WaterType {
        enum Hot;
        enum Cold;
    }

    port def Spigot {
        attribute waterType : WaterType;
    }

    port def FaucetInlet {
        attribute waterType : WaterType;
    }

    part def SpigotBank {
        port hotOutlet : Spigot;
        port coldOutlet : Spigot;
        attribute hotWaterType : WaterType = WaterType::Hot;
        attribute coldWaterType : WaterType = WaterType::Cold;
    }

    part def Faucet {
        port hotInlet : FaucetInlet;
        port coldInlet : FaucetInlet;
        attribute hotWaterType : WaterType = WaterType::Hot;
        attribute coldWaterType : WaterType = WaterType::Cold;
    }

    part def WaterSupplySystem {
        part mainSupply : SpigotBank;
        part faucet1 : Faucet;
        part faucet2 : Faucet;

        connect mainSupply.hotOutlet to faucet1.hotInlet;
        connect mainSupply.coldOutlet to faucet1.coldInlet;
        connect mainSupply.hotOutlet to faucet2.hotInlet;
        connect mainSupply.coldOutlet to faucet2.coldInlet;
    }
}