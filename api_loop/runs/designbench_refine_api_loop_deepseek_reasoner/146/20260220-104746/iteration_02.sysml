package AscentTrajectorySystem {
    private import ScalarValues::*;
    
    item def Vector3 {
        attribute x : Real;
        attribute y : Real;
        attribute z : Real;
    }
    
    item def SamplingPoint {
        attribute time : Real;
        attribute position : Vector3;
        attribute velocity : Vector3;
    }
    
    part def TrajectorySampler {
        attribute missionLiftoffUTC : String;
        attribute ascentData : SamplingPoint[*];
    }
    
    part def Spacecraft {
        port statePort : StatePort;
    }
    
    port def StatePort {
        attribute currentTime : Real;
        attribute currentPosition : Vector3;
        attribute currentVelocity : Vector3;
    }
    
    part trajectorySampler : TrajectorySampler;
    part spacecraft : Spacecraft;
    
    connect spacecraft.statePort to trajectorySampler;
}