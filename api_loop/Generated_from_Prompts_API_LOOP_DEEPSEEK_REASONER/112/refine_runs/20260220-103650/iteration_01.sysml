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
        port in startIn : StartSignal;
        port out startOut : StartSignal;
        port in vehicleOnIn : VehicleOnSignal;
        port out vehicleOnOut : VehicleOnSignal;
        port in overTempIn : OverTemperatureSignal;
        port out overTempOut : OverTemperatureSignal;
        port in restoreIn : RestoreNormalSignal;
        port out restoreOut : RestoreNormalSignal;
        port in brakeIn : BrakePressedSignal;
    }

    part def Vehicle {
        port in startIn : StartSignal;
        port in vehicleOnIn : VehicleOnSignal;
        port out overTempOut : OverTemperatureSignal;
        port in restoreIn : RestoreNormalSignal;
        port in brakeIn : BrakePressedSignal;
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