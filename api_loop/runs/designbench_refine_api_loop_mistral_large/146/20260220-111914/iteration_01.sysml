package SpacecraftTrajectory {

    private import ScalarValues::*;

    enum def TimeReference {
        enum MissionLiftoffTime
    }

    part def Spacecraft {
        attribute liftoffUTC : String = "22:42:32.924534 on August 23, 2020";
        attribute referenceTimeZero : Real = 0.0;
        attribute timePoints : Sequence<Real>;
        attribute positions : Sequence<Vector3D>;
        attribute velocities : Sequence<Vector3D>;
    }

    part def Vector3D {
        attribute x : Real;
        attribute y : Real;
        attribute z : Real;
    }

    part def TrajectorySampler {
        attribute sampleInterval : Real = 1.0;
        port dataOut : TrajectoryDataPort;
    }

    port def TrajectoryDataPort {
        attribute time : Real;
        attribute position : Vector3D;
        attribute velocity : Vector3D;
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