package VehicleStateManagement {
    private import ScalarValues::*;
    
    enum def VehicleState {
        enum off;
        enum starting;
        enum on;
    }
    
    part def VehicleSystem {
        attribute currentState : VehicleState = VehicleState::off;
        
        port def ControlPort {
            attribute receivedSignal : String;
        }
        
        port controlInput : in ControlPort;
    }
    
    part vehicleController : VehicleSystem;
}