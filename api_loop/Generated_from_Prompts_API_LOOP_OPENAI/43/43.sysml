package VehicleStartStop {
  private import ScalarValues::*;

  enum def VehicleState { Off; Starting; Started; Shutdown; }

  port def SignalPort { }
  port def PowerPort { }

  part def VehicleController {
    port startControlIn : SignalPort;
    port powerIn : PowerPort;
  }

  part def DriverInterface {
    port startSignalOut : SignalPort;
    port brakeSignalOut : SignalPort;
    port vehicleOnSignalOut : SignalPort;
    port shutdownSignalOut : SignalPort;
  }

  part def ParkingBrake {
    port applyCommandIn : SignalPort;
    attribute applied : Boolean = false;
  }

  part def VehiclePower {
    port powerOut : PowerPort;
    attribute enabled : Boolean = false;
  }

  part def StartStopSystem {
    attribute vehicleState : VehicleState = VehicleState::Off;
    port startSignalIn : SignalPort;
    port brakeSignalIn : SignalPort;
    port vehicleOnSignalIn : SignalPort;
    port shutdownSignalIn : SignalPort;
    port startControlOut : SignalPort;
    port parkingBrakeApplyOut : SignalPort;
    port powerSupplyOut : PowerPort;
  }

  part def Vehicle {
    part controller : VehicleController;
    part power : VehiclePower;
    part parkingBrake : ParkingBrake;
  }

  part def VehicleStartStopContext {
    part system : StartStopSystem;
    part driver : DriverInterface;
    part vehicle : Vehicle;

    connect driver.startSignalOut to system.startSignalIn;
    connect driver.brakeSignalOut to system.brakeSignalIn;
    connect driver.vehicleOnSignalOut to system.vehicleOnSignalIn;
    connect driver.shutdownSignalOut to system.shutdownSignalIn;

    connect system.startControlOut to vehicle.controller.startControlIn;
    connect system.powerSupplyOut to vehicle.power.powerOut;
    connect system.parkingBrakeApplyOut to vehicle.parkingBrake.applyCommandIn;
    connect vehicle.power.powerOut to vehicle.controller.powerIn;
  }
}