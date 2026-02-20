package SpacecraftTrajectory {

    private import ScalarValues::*;

    part def Vector3D {
        attribute x : Real;
        attribute y : Real;
        attribute z : Real;
    }

    enum def TimeReference {
        enum MissionLiftoffTime;
    }

    part def Spacecraft {
        attribute liftoffUTC : String = "22:42:32.924534 on August 23, 2020";
        attribute referenceTimeZero : ScalarValues::Real = 0.0;
        attribute timePoints : ScalarValues::Real[0..*];
        attribute positions : Vector3D[0..*];
        attribute velocities : Vector3D[0..*];
    }

    port def TrajectoryDataPort {
        attribute time : ScalarValues::Real;
        attribute position : Vector3D;
        attribute velocity : Vector3D;
    }

    part def TrajectorySampler {
        attribute sampleInterval : ScalarValues::Real = 1.0;
        port dataOut : TrajectoryDataPort;
    }

    part def TrajectoryRecorder {
        port dataIn : TrajectoryDataPort;
    }

    part system {
        part spacecraft : Spacecraft;
        part sampler : TrajectorySampler;
        part recorder : TrajectoryRecorder;

        connect sampler.dataOut to recorder.dataIn;
    }
}