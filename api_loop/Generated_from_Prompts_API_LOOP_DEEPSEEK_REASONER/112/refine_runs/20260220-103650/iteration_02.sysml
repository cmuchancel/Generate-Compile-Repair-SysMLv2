package VehicleManagementSystem {
    private import ScalarValues::*;

    enum def VehicleState {
        enum Off;
        enum Starting;
        enum Running;
        enum HealthManagement;
        enum Maintenance;
        enum Degraded;
        enum Normal;
    }

    item def StartSignal {}
    item def VehicleOnSignal {}
    item def OverTemperatureSignal {}
    item def RestoreNormalSignal {}
    item def BrakePressedSignal {}

    part def VehicleController {
        port startIn : StartSignal;
        port startOut : StartSignal;
        port vehicleOnIn : VehicleOnSignal;
        port vehicleOnOut : VehicleOnSignal;
        port overTempIn : OverTemperatureSignal;
        port overTempOut : OverTemperatureSignal;
        port restoreIn : RestoreNormalSignal;
        port restoreOut : RestoreNormalSignal;
        port brakeIn : BrakePressedSignal;
    }

    part def Vehicle {
        port startIn : StartSignal;
        port vehicleOnIn : VehicleOnSignal;
        port overTempOut : OverTemperatureSignal;
        port restoreIn : RestoreNormalSignal;
        port brakeIn : BrakePressedSignal;
        attribute currentState : VehicleState = VehicleState::Off;
        attribute temperature : Real;
    }

    part controller : VehicleController;
    part vehicle : Vehicle;

    connect controller.startOut to vehicle.startIn;
    connect controller.vehicleOnOut to vehicle.vehicleOnIn;
    connect vehicle.overTempOut to controller.overTempIn;
    connect controller.restoreOut to vehicle.restoreIn;
    connect controller.brakeIn to vehicle.brakeIn;
}