package TrajectoryRecordingSystem {
    private import ScalarValues::*;

    attribute def CartesianPosition {
        attribute x : Real;
        attribute y : Real;
        attribute z : Real;
    }

    attribute def CartesianVelocity {
        attribute vx : Real;
        attribute vy : Real;
        attribute vz : Real;
    }

    attribute def TrajectorySample {
        attribute timePoint : Real;
        attribute position : CartesianPosition;
        attribute velocity : CartesianVelocity;
    }

    part def LiftoffTimeReference {
        attribute utcYear : Integer = 2020;
        attribute utcMonth : Integer = 8;
        attribute utcDay : Integer = 23;
        attribute utcHour : Integer = 22;
        attribute utcMinute : Integer = 42;
        attribute utcSecond : Real = 32.924534;
        attribute referenceTimeZero : Real = 0.0;
    }

    part def AscentTrajectoryRecorder {
        attribute missionName : String;
        attribute sample0 : TrajectorySample;
        attribute sample1 : TrajectorySample;
        attribute sample2 : TrajectorySample;
        attribute sample3 : TrajectorySample;
        attribute sample4 : TrajectorySample;
        attribute sample5 : TrajectorySample;
        attribute sample6 : TrajectorySample;
        attribute sample7 : TrajectorySample;
        attribute sample8 : TrajectorySample;
        attribute sample9 : TrajectorySample;
        part liftoffRef : LiftoffTimeReference;
    }

    part def SpacecraftAscentSystem {
        part recorder : AscentTrajectoryRecorder;
    }

    part ascentSystem : SpacecraftAscentSystem;
}