package WaterSupplyNetwork {
    private import ScalarValues::*;

    port def HotWaterPort;
    port def ColdWaterPort;

    part def SpigotBank {
        port hotSupply : HotWaterPort;
        port coldSupply : ColdWaterPort;
    }

    part def Faucet {
        port hotInlet : HotWaterPort;
        port coldInlet : ColdWaterPort;
    }

    item def WaterSupplySystem {
        part mainSupply : SpigotBank;
        part faucet1 : Faucet;
        part faucet2 : Faucet;

        connect mainSupply.hotSupply to faucet1.hotInlet;
        connect mainSupply.coldSupply to faucet1.coldInlet;
        connect mainSupply.hotSupply to faucet2.hotInlet;
        connect mainSupply.coldSupply to faucet2.coldInlet;
    }
}