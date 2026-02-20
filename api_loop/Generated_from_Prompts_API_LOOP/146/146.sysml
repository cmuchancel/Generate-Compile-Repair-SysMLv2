package TrajectorySampling {
    private import ScalarValues::*;

    item def Vector3 {
        attribute x_m : Real;
        attribute y_m : Real;
        attribute z_m : Real;
    }

    item def TrajectorySample {
        attribute time_s : Real;
        attribute position : Vector3;
        attribute velocity : Vector3;
    }

    part def TrajectoryRecorder {
        attribute liftoffUTC : String;
        attribute liftoffTimeZero_s : Real = 0.0;
        attribute samples : TrajectorySample[*];
    }

    part def SpacecraftTrajectorySystem {
        part recorder : TrajectoryRecorder;
    }

    part system : SpacecraftTrajectorySystem;
}