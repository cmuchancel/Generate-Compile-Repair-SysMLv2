package AutomobileSystem {
    private import ScalarValues::*;
    
    enum def FuelStatus {
        enum fully_fueled;
        enum empty;
    }
    
    port def DriveInterface;
    
    part def Engine {
        attribute torque_output : Real;
        port drive_out : DriveInterface;
    }
    
    part def Transmission {
        port drive_in : DriveInterface;
    }
    
    item def MassRequirement {
        attribute mass_grade : Real;
        attribute max_mass : Real;
    }
    
    item def FuelRequirement {
        attribute required_status : FuelStatus;
    }
    
    item def PowertrainRequirement {
        attribute min_torque : Real;
    }
    
    item def ReliabilityRequirement {
        attribute life_cycle_years : Integer;
    }
    
    part def Vehicle {
        attribute mass : Real;
        attribute actual_mass : Real;
        attribute fuel_state : FuelStatus;
        
        part engine : Engine;
        part transmission : Transmission;
        
        port powertrain_if : DriveInterface;
        
        connect engine.drive_out to transmission.drive_in;
        connect engine.drive_out to powertrain_if;
    }
    
    part LightVehicle : Vehicle {
        attribute mass : Real = 2000;
        attribute actual_mass : Real = 1900;
        attribute fuel_state : FuelStatus = FuelStatus::fully_fueled;
        
        part engine : Engine {
            attribute torque_output : Real = 200;
        }
    }
    
    part HeavyVehicle : Vehicle {
        attribute mass : Real = 2500;
        attribute actual_mass : Real = 2400;
        attribute fuel_state : FuelStatus = FuelStatus::empty;
        
        part engine : Engine {
            attribute torque_output : Real = 350;
        }
    }
    
    item MassReq2000 : MassRequirement {
        attribute mass_grade : Real = 2000;
        attribute max_mass : Real = 2200;
    }
    
    item MassReq2500 : MassRequirement {
        attribute mass_grade : Real = 2500;
        attribute max_mass : Real = 2700;
    }
    
    item FuelReqFull : FuelRequirement {
        attribute required_status : FuelStatus = FuelStatus::fully_fueled;
    }
    
    item FuelReqEmpty : FuelRequirement {
        attribute required_status : FuelStatus = FuelStatus::empty;
    }
    
    item PowertrainReqStandard : PowertrainRequirement {
        attribute min_torque : Real = 180;
    }
    
    item ReliabilityReq5Year : ReliabilityRequirement {
        attribute life_cycle_years : Integer = 5;
    }
}