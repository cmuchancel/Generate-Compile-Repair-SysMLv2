package VehicleStartStop {
   private import ScalarValues::*;
   enum def VehicleState {
      enum shutdown;
      enum startup;
      enum running;
   }
   part def VehicleControlSystem {
      attribute vehicleStatus : VehicleState = VehicleState::shutdown;
      attribute startSignal : Boolean = false;
      attribute startedSignal : Boolean = false;
      attribute shutdownSignal : Boolean = false;
   }
   part vehicleControlSystem : VehicleControlSystem;
}