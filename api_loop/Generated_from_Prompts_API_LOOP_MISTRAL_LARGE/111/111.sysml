package AutomobileAssembly {
    private import ScalarValues::*;

    part def VehicleAssembly;
    part def Gearbox;
    part def Engine;
    part def FullyAssembledVehicle;

    part def AssemblyLine {
        part vehicleAssembly : VehicleAssembly;
        part gearbox : Gearbox;
        part engine : Engine;
        part assemblyWithoutEngine : VehicleAssembly;
        part fullyAssembledVehicle : FullyAssembledVehicle;

        attribute gearboxInstalled : Boolean = false;
        attribute engineInstalled : Boolean = false;
    }

    part assemblyProcess : AssemblyLine;

    connect assemblyProcess.vehicleAssembly to assemblyProcess.gearbox;
    connect assemblyProcess.gearbox to assemblyProcess.assemblyWithoutEngine;
    connect assemblyProcess.assemblyWithoutEngine to assemblyProcess.engine;
    connect assemblyProcess.engine to assemblyProcess.fullyAssembledVehicle;
}