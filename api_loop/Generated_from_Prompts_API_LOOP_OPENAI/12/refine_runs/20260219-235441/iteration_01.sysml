package WaterSupplyModel
    private import ScalarValues::*;
    
    item def HotWater;
    item def ColdWater;
    
    port def HotPort {
        item hot: HotWater;
    }
    
    port def ColdPort {
        item cold: ColdWater;
    }
    
    part def SpigotBank {
        port hotOut: HotPort;
        port coldOut: ColdPort;
    }
    
    part def Spigot {
        port hotIn: HotPort;
        port coldIn: ColdPort;
        port hotOut: HotPort;
        port coldOut: ColdPort;
        connect hotIn to hotOut;
        connect coldIn to coldOut;
    }
    
    part def FaucetInlet {
        port hotIn: HotPort;
        port coldIn: ColdPort;
    }
    
    part def Faucet {
        part inlet: FaucetInlet;
    }
    
    part def WaterSupplySystem {
        part bank: SpigotBank;
        part spigot1: Spigot;
        part spigot2: Spigot;
        part faucet1: Faucet;
        part faucet2: Faucet;
        
        connect bank.hotOut to spigot1.hotIn;
        connect bank.coldOut to spigot1.coldIn;
        connect bank.hotOut to spigot2.hotIn;
        connect bank.coldOut to spigot2.coldIn;
        
        connect spigot1.hotOut to faucet1.inlet.hotIn;
        connect spigot1.coldOut to faucet1.inlet.coldIn;
        
        connect spigot2.hotOut to faucet2.inlet.hotIn;
        connect spigot2.coldOut to faucet2.inlet.coldIn;
    }
end package