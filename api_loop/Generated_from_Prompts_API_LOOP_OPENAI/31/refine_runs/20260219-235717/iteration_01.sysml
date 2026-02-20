package VehicleSimulation {
  private import ScalarValues::*;
  
  part def VehicleSim {
    attribute mass : Real;
    attribute initialPosition : Real;
    attribute initialVelocity : Real;
    attribute timeStep : Real;
  }
  
  part def PowerInput {
    attribute time : Real;
    attribute power : Real;
  }
  
  part def StepResult {
    attribute time : Real;
    attribute inputPower : Real;
    attribute velocity : Real;
    attribute position : Real;
  }
  
  part def Trajectory {
    part step1 : StepResult;
    part step2 : StepResult;
    part step3 : StepResult;
  }
  
  part def Simulation {
    part vehicle : VehicleSim;
    part input1 : PowerInput;
    part input2 : PowerInput;
    part input3 : PowerInput;
    part trajectory : Trajectory;
  }
}