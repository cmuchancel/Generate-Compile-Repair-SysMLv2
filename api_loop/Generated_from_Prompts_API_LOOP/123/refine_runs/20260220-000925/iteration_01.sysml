package VehicleLongitudinalSimulation {
  private import ScalarValues::*;
  
  item def SegmentData {
    attribute timeStart : Real;
    attribute timeEnd : Real;
    attribute power : Real;
  }
  
  item def SequenceData {
    attribute value1 : Real;
    attribute value2 : Real;
    attribute value3 : Real;
    attribute value4 : Real;
    attribute value5 : Real;
  }
  
  part def VehicleInputs {
    attribute vehicleMass : Real;
    attribute initialPosition : Real;
    attribute initialVelocity : Real;
    attribute timeStep : Real;
    attribute segment1TimeStart : Real;
    attribute segment1TimeEnd : Real;
    attribute segment1Power : Real;
    attribute segment2TimeStart : Real;
    attribute segment2TimeEnd : Real;
    attribute segment2Power : Real;
    attribute segment3TimeStart : Real;
    attribute segment3TimeEnd : Real;
    attribute segment3Power : Real;
  }
  
  part def VehicleOutputs {
    attribute accelerationSequence1 : Real;
    attribute accelerationSequence2 : Real;
    attribute accelerationSequence3 : Real;
    attribute accelerationSequence4 : Real;
    attribute accelerationSequence5 : Real;
    attribute velocitySequence1 : Real;
    attribute velocitySequence2 : Real;
    attribute velocitySequence3 : Real;
    attribute velocitySequence4 : Real;
    attribute velocitySequence5 : Real;
    attribute positionSequence1 : Real;
    attribute positionSequence2 : Real;
    attribute positionSequence3 : Real;
    attribute positionSequence4 : Real;
    attribute positionSequence5 : Real;
  }
  
  part def LongitudinalDynamicsSimulator {
    part inputs : VehicleInputs;
    part outputs : VehicleOutputs;
  }
  
  part Simulator : LongitudinalDynamicsSimulator;
}