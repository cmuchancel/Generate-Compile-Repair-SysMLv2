package VehicleSafetySecurity {
    private import ScalarValues::*;
    
    part def AlarmSystem;
    part def SeatBelt;
    part def FrontRowSeat;
    part def DriversAirbag;
    part def VehicleBody;
    part def Bumper;
    part def KeylessEntrySystem;
    
    part def Interior {
        part alarm : AlarmSystem;
        part seatBelt : SeatBelt;
        part frontRowSeat : FrontRowSeat;
        part driversAirbag : DriversAirbag;
    }
    
    part def BodyAssembly {
        part vehicleBody : VehicleBody;
        part bumper : Bumper;
        part keylessEntry : KeylessEntrySystem;
    }
    
    part def VehicleSafetySecuritySystem {
        part interior : Interior;
        part bodyAssembly : BodyAssembly;
    }
}