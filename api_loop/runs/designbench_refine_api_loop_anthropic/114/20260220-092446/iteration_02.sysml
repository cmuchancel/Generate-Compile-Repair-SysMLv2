package VehicleSpatiotemporalSimulation {
    private import ScalarValues::*;

    enum def VehicleState {
        enum on;
        enum off;
    }

    enum def RoadSurface {
        enum dry;
        enum wet;
        enum icy;
    }

    part def Road {
        attribute slopeAngle : Real;
        attribute frictionCoefficient : Real;
        attribute surfaceType : RoadSurface;
    }

    part def Vehicle {
        attribute mass : Real;
        attribute positionX : Real;
        attribute positionY : Real;
        attribute positionZ : Real;
        attribute velocityX : Real;
        attribute velocityY : Real;
        attribute velocityZ : Real;
        attribute accelerationX : Real;
        attribute accelerationY : Real;
        attribute accelerationZ : Real;
        attribute vehicleState : VehicleState;
    }

    part def SimulationSnapshot {
        attribute timePoint : Real;
        attribute snapshotPositionX : Real;
        attribute snapshotPositionY : Real;
        attribute snapshotPositionZ : Real;
        attribute snapshotVelocityX : Real;
        attribute snapshotVelocityY : Real;
        attribute snapshotVelocityZ : Real;
        attribute snapshotAccelerationX : Real;
        attribute snapshotAccelerationY : Real;
        attribute snapshotAccelerationZ : Real;
        attribute snapshotVehicleState : VehicleState;
        attribute snapshotSlopeAngle : Real;
        attribute snapshotFriction : Real;
    }

    part def SimulationTimeSeries {
        attribute startTime : Real;
        attribute endTime : Real;
        attribute timeStepCount : Integer;
        part snapshot1 : SimulationSnapshot;
        part snapshot2 : SimulationSnapshot;
        part snapshot3 : SimulationSnapshot;
        part snapshot4 : SimulationSnapshot;
        part snapshot5 : SimulationSnapshot;
    }

    port def VehicleDataPort {
        attribute portMass : Real;
        attribute portPositionX : Real;
        attribute portPositionY : Real;
        attribute portVelocityX : Real;
        attribute portVelocityY : Real;
        attribute portVehicleState : VehicleState;
    }

    port def RoadDataPort {
        attribute portSlopeAngle : Real;
        attribute portFriction : Real;
        attribute portSurface : RoadSurface;
    }

    part def SimulationSystem {
        attribute simulationId : Integer;
        attribute description : String;

        part road : Road;
        part vehicle : Vehicle;
        part timeSeries : SimulationTimeSeries;

        port vehiclePort : VehicleDataPort;
        port roadPort : RoadDataPort;
    }

    part def DynamicsAnalyzer {
        attribute analyzerId : Integer;
        port inputVehicle : VehicleDataPort;
        port inputRoad : RoadDataPort;
        attribute analysisResult : Real;
        attribute performanceScore : Real;
    }

    part def RoadAdaptabilityAssessor {
        attribute assessorId : Integer;
        port vehicleInput : VehicleDataPort;
        port roadInput : RoadDataPort;
        attribute adaptabilityScore : Real;
        attribute predictedPerformance : Real;
    }

    part simulationContext : SimulationSystem {
        attribute :>> simulationId = 1;
        attribute :>> description = "Vehicle Spatiotemporal Simulation";

        part :>> road {
            attribute :>> slopeAngle = 5.0;
            attribute :>> frictionCoefficient = 0.7;
            attribute :>> surfaceType = RoadSurface::dry;
        }

        part :>> vehicle {
            attribute :>> mass = 1500.0;
            attribute :>> positionX = 0.0;
            attribute :>> positionY = 0.0;
            attribute :>> positionZ = 0.0;
            attribute :>> velocityX = 0.0;
            attribute :>> velocityY = 0.0;
            attribute :>> velocityZ = 0.0;
            attribute :>> accelerationX = 0.0;
            attribute :>> accelerationY = 0.0;
            attribute :>> accelerationZ = 0.0;
            attribute :>> vehicleState = VehicleState::off;
        }

        part :>> timeSeries {
            attribute :>> startTime = 0.0;
            attribute :>> endTime = 100.0;
            attribute :>> timeStepCount = 5;

            part :>> snapshot1 {
                attribute :>> timePoint = 0.0;
                attribute :>> snapshotPositionX = 0.0;
                attribute :>> snapshotPositionY = 0.0;
                attribute :>> snapshotPositionZ = 0.0;
                attribute :>> snapshotVelocityX = 0.0;
                attribute :>> snapshotVelocityY = 0.0;
                attribute :>> snapshotVelocityZ = 0.0;
                attribute :>> snapshotAccelerationX = 0.0;
                attribute :>> snapshotAccelerationY = 0.0;
                attribute :>> snapshotAccelerationZ = 0.0;
                attribute :>> snapshotVehicleState = VehicleState::on;
                attribute :>> snapshotSlopeAngle = 5.0;
                attribute :>> snapshotFriction = 0.7;
            }

            part :>> snapshot2 {
                attribute :>> timePoint = 25.0;
                attribute :>> snapshotPositionX = 100.0;
                attribute :>> snapshotPositionY = 0.0;
                attribute :>> snapshotPositionZ = 0.0;
                attribute :>> snapshotVelocityX = 15.0;
                attribute :>> snapshotVelocityY = 0.0;
                attribute :>> snapshotVelocityZ = 0.0;
                attribute :>> snapshotAccelerationX = 2.0;
                attribute :>> snapshotAccelerationY = 0.0;
                attribute :>> snapshotAccelerationZ = 0.0;
                attribute :>> snapshotVehicleState = VehicleState::on;
                attribute :>> snapshotSlopeAngle = 5.0;
                attribute :>> snapshotFriction = 0.7;
            }

            part :>> snapshot3 {
                attribute :>> timePoint = 50.0;
                attribute :>> snapshotPositionX = 500.0;
                attribute :>> snapshotPositionY = 0.0;
                attribute :>> snapshotPositionZ = 0.0;
                attribute :>> snapshotVelocityX = 30.0;
                attribute :>> snapshotVelocityY = 0.0;
                attribute :>> snapshotVelocityZ = 0.0;
                attribute :>> snapshotAccelerationX = 0.0;
                attribute :>> snapshotAccelerationY = 0.0;
                attribute :>> snapshotAccelerationZ = 0.0;
                attribute :>> snapshotVehicleState = VehicleState::on;
                attribute :>> snapshotSlopeAngle = 5.0;
                attribute :>> snapshotFriction = 0.7;
            }

            part :>> snapshot4 {
                attribute :>> timePoint = 75.0;
                attribute :>> snapshotPositionX = 1200.0;
                attribute :>> snapshotPositionY = 0.0;
                attribute :>> snapshotPositionZ = 0.0;
                attribute :>> snapshotVelocityX = 10.0;
                attribute :>> snapshotVelocityY = 0.0;
                attribute :>> snapshotVelocityZ = 0.0;
                attribute :>> snapshotAccelerationX = -3.0;
                attribute :>> snapshotAccelerationY = 0.0;
                attribute :>> snapshotAccelerationZ = 0.0;
                attribute :>> snapshotVehicleState = VehicleState::on;
                attribute :>> snapshotSlopeAngle = 5.0;
                attribute :>> snapshotFriction = 0.7;
            }

            part :>> snapshot5 {
                attribute :>> timePoint = 100.0;
                attribute :>> snapshotPositionX = 1500.0;
                attribute :>> snapshotPositionY = 0.0;
                attribute :>> snapshotPositionZ = 0.0;
                attribute :>> snapshotVelocityX = 0.0;
                attribute :>> snapshotVelocityY = 0.0;
                attribute :>> snapshotVelocityZ = 0.0;
                attribute :>> snapshotAccelerationX = 0.0;
                attribute :>> snapshotAccelerationY = 0.0;
                attribute :>> snapshotAccelerationZ = 0.0;
                attribute :>> snapshotVehicleState = VehicleState::off;
                attribute :>> snapshotSlopeAngle = 5.0;
                attribute :>> snapshotFriction = 0.7;
            }
        }
    }

    part analyzer : DynamicsAnalyzer {
        attribute :>> analyzerId = 1;
        attribute :>> analysisResult = 0.0;
        attribute :>> performanceScore = 0.0;
    }

    part assessor : RoadAdaptabilityAssessor {
        attribute :>> assessorId = 1;
        attribute :>> adaptabilityScore = 0.0;
        attribute :>> predictedPerformance = 0.0;
    }

    connect simulationContext.vehiclePort to analyzer.inputVehicle;
    connect simulationContext.roadPort to analyzer.inputRoad;
    connect simulationContext.vehiclePort to assessor.vehicleInput;
    connect simulationContext.roadPort to assessor.roadInput;
}