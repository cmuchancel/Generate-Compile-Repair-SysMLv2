package TrajectoryRecordingSystem {
    private import ScalarValues::*;

    attribute def TimeReference {
        attribute missionElapsedSeconds : Real;
        attribute utcYear : Integer;
        attribute utcMonth : Integer;
        attribute utcDay : Integer;
        attribute utcHour : Integer;
        attribute utcMinute : Integer;
        attribute utcSecond : Real;
    }

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

    attribute def Trajectorysample {
        attribute timePoint : Real;
        attribute position : CartesianPosition;
        attribute velocity : CartesianVelocity;
    }

    part def LiftoffTimeReference {
        attribute utcHour : Integer = 22;
        attribute utcMinute : Integer = 42;
        attribute utcSecond : Real = 32.924534;
        attribute utcYear : Integer = 2020;
        attribute utcMonth : Integer = 8;
        attribute utcDay : Integer = 23;
        attribute referenceTimeZero : Real = 0.0;
    }

    part def AscentTrajectoryRecorder {
        attribute missionName : String;

        attribute sample0 : Trajectorysample;
        attribute sample1 : Trajectorysample;
        attribute sample2 : Trajectorysample;
        attribute sample3 : Trajectorysample;
        attribute sample4 : Trajectorysample;
        attribute sample5 : Trajectorysample;
        attribute sample6 : Trajectorysample;
        attribute sample7 : Trajectoryample;
        attribute sample8 : Trajectoryample;
        attribute sample9 : Trajectoryample;

        part liftoffRef : LiftoffTimeReference;
    }

    part def SpacecraftAscentSystem {
        part recorder : AscentTrajectoryRecorder;
    }

    part ascentSystem : SpacecraftAscentSystem {
        part recorder : AscentTrajectoryRecorder {
            attribute missionName : String = "AscentMission_20200823";

            attribute sample0 : Trajectoryample = {
                timePoint = 0.0;
                position = { x = 0.0; y = 0.0; z = 0.0; };
                velocity = { vx = 0.0; vy = 0.0; vz = 0.0; };
            };

            attribute sample1 : Trajectoryample = {
                timePoint = 10.0;
                position = { x = 0.0; y = 0.0; z = 50.0; };
                velocity = { vx = 0.0; vy = 0.0; vz = 10.0; };
            };

            attribute sample2 : Trajectoryample = {
                timePoint = 20.0;
                position = { x = 0.0; y = 0.0; z = 200.0; };
                velocity = { vx = 0.0; vy = 0.0; vz = 30.0; };
            };

            attribute sample3 : Trajectoryample = {
                timePoint = 30.0;
                position = { x = 10.0; y = 5.0; z = 500.0; };
                velocity = { vx = 1.0; vy = 0.5; vz = 60.0; };
            };

            attribute sample4 : Trajectoryample = {
                timePoint = 60.0;
                position = { x = 50.0; y = 25.0; z = 2000.0; };
                velocity = { vx = 5.0; vy = 2.5; vz = 150.0; };
            };

            attribute sample5 : Trajectoryample = {
                timePoint = 120.0;
                position = { x = 200.0; y = 100.0; z = 8000.0; };
                velocity = { vx = 20.0; vy = 10.0; vz = 400.0; };
            };

            attribute sample6 : Trajectoryample = {
                timePoint = 180.0;
                position = { x = 500.0; y = 250.0; z = 20000.0; };
                velocity = { vx = 50.0; vy = 25.0; vz = 900.0; };
            };

            attribute sample7 : Trajectoryample = {
                timePoint = 240.0;
                position = { x = 1000.0; y = 500.0; z = 40000.0; };
                velocity = { vx = 100.0; vy = 50.0; vz = 1500.0; };
            };

            attribute sample8 : Trajectoryample = {
                timePoint = 300.0;
                position = { x = 2000.0; y = 1000.0; z = 70000.0; };
                velocity = { vx = 200.0; vy = 100.0; vz = 2200.0; };
            };

            attribute sample9 : Trajectoryample = {
                timePoint = 360.0;
                position = { x = 3500.0; y = 1750.0; z = 110000.0; };
                velocity = { vx = 350.0; vy = 175.0; vz = 3000.0; };
            };

            part liftoffRef : LiftoffTimeReference;
        }
    }
}