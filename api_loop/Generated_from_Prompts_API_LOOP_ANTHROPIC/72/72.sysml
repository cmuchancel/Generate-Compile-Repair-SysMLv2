package VehicleDynamicsAnalysis {
    private import ScalarValues::*;

    enum def FuelType {
        enum Gasoline;
        enum Diesel;
        enum Electric;
        enum Hybrid;
    }

    enum def DriveMode {
        enum Urban;
        enum Highway;
        enum Mixed;
    }

    enum def ComplianceStatus {
        enum Pass;
        enum Fail;
        enum Pending;
    }

    part def VehicleAttributes {
        attribute curbWeight : Real;
        attribute payload : Real;
        attribute totalMass : Real;
        attribute wheelDiameter : Real;
        attribute transmissionEfficiency : Real;
        attribute fuelEconomyUrban : Real;
        attribute fuelEconomyHighway : Real;
        attribute fuelEconomyMixed : Real;
        attribute dragCoefficient : Real;
        attribute frontalArea : Real;
        attribute rollingResistanceCoefficient : Real;
        attribute fuelType : FuelType;
    }

    part def DriveCyclePoint {
        attribute timeStamp : Real;
        attribute position : Real;
        attribute speed : Real;
        attribute acceleration : Real;
    }

    part def DriveCycle {
        attribute cycleName : String;
        attribute totalDuration : Real;
        attribute totalDistance : Real;
        attribute driveMode : DriveMode;
        attribute averageSpeed : Real;
        attribute maxSpeed : Real;
        attribute minSpeed : Real;
        part point1 : DriveCyclePoint;
        part point2 : DriveCyclePoint;
        part point3 : DriveCyclePoint;
        part point4 : DriveCyclePoint;
        part point5 : DriveCyclePoint;
        part point6 : DriveCyclePoint;
        part point7 : DriveCyclePoint;
        part point8 : DriveCyclePoint;
        part point9 : DriveCyclePoint;
        part point10 : DriveCyclePoint;
    }

    part def PowerDemandCalculator {
        attribute gravitationalAcceleration : Real;
        attribute airDensity : Real;
        attribute calculatedTractivePower : Real;
        attribute aerodynamicDragPower : Real;
        attribute rollingResistancePower : Real;
        attribute gradePower : Real;
        attribute inertiaPower : Real;
        attribute totalPowerDemand : Real;
        attribute enginePowerRequired : Real;
    }

    part def AccelerationCalculator {
        attribute currentSpeed : Real;
        attribute previousSpeed : Real;
        attribute timeInterval : Real;
        attribute calculatedAcceleration : Real;
        attribute maxAcceleration : Real;
        attribute minAcceleration : Real;
        attribute averageAcceleration : Real;
    }

    part def SpeedPositionTracker {
        attribute currentPosition : Real;
        attribute currentSpeed : Real;
        attribute elapsedTime : Real;
        attribute distanceTraveled : Real;
        attribute speedAtTime : Real;
        attribute positionAtTime : Real;
    }

    part def FuelEconomyCalculator {
        attribute totalFuelConsumed : Real;
        attribute totalDistanceCovered : Real;
        attribute fuelEconomyMPG : Real;
        attribute fuelEconomyL100km : Real;
        attribute urbanFuelConsumption : Real;
        attribute highwayFuelConsumption : Real;
        attribute mixedFuelConsumption : Real;
        attribute energyConsumptionRate : Real;
    }

    part def ComplianceVerifier {
        attribute targetFuelEconomyMPG : Real;
        attribute targetFuelEconomyL100km : Real;
        attribute actualFuelEconomyMPG : Real;
        attribute actualFuelEconomyL100km : Real;
        attribute complianceStatus : ComplianceStatus;
        attribute marginMPG : Real;
        attribute marginL100km : Real;
        attribute isCompliant : Boolean;
        attribute verificationNotes : String;
    }

    part def ConfigurationManager {
        attribute configurationName : String;
        attribute isActive : Boolean;
        attribute lastUpdated : String;
        attribute numberOfScenarios : Integer;
        attribute selectedScenarioIndex : Integer;
    }

    part def PerformanceAnalysisResult {
        attribute analysisId : String;
        attribute analysisTimestamp : String;
        attribute peakPowerDemand : Real;
        attribute averagePowerDemand : Real;
        attribute peakAcceleration : Real;
        attribute peakDeceleration : Real;
        attribute topSpeedAchieved : Real;
        attribute totalEnergyConsumed : Real;
        attribute overallFuelEconomyMPG : Real;
        attribute overallFuelEconomyL100km : Real;
        attribute complianceResult : ComplianceStatus;
    }

    port def VehicleDataPort {
        attribute curbWeight : Real;
        attribute payload : Real;
        attribute wheelDiameter : Real;
        attribute transmissionEfficiency : Real;
    }

    port def DriveCyclePort {
        attribute cycleName : String;
        attribute totalDuration : Real;
        attribute totalDistance : Real;
    }

    port def PowerDataPort {
        attribute totalPowerDemand : Real;
        attribute enginePowerRequired : Real;
    }

    port def FuelEconomyPort {
        attribute fuelEconomyMPG : Real;
        attribute fuelEconomyL100km : Real;
        attribute totalFuelConsumed : Real;
    }

    port def CompliancePort {
        attribute isCompliant : Boolean;
        attribute complianceStatus : ComplianceStatus;
        attribute marginMPG : Real;
    }

    port def ResultsPort {
        attribute analysisId : String;
        attribute overallFuelEconomyMPG : Real;
        attribute complianceResult : ComplianceStatus;
    }

    part def VehicleDynamicsSystem {
        attribute systemName : String;
        attribute systemVersion : String;
        attribute fuelEconomyTarget30MPG : Real;
        attribute targetFuelConsumptionL100km : Real;

        part vehicleAttributes : VehicleAttributes;
        part driveCycleUrban : DriveCycle;
        part driveCycleHighway : DriveCycle;
        part driveCycleMixed : DriveCycle;
        part powerCalculator : PowerDemandCalculator;
        part accelerationCalc : AccelerationCalculator;
        part speedPositionTracker : SpeedPositionTracker;
        part fuelEconomyCalc : FuelEconomyCalculator;
        part complianceVerifier : ComplianceVerifier;
        part configManager : ConfigurationManager;
        part analysisResult : PerformanceAnalysisResult;

        port vehicleDataIn : VehicleDataPort;
        port driveCycleIn : DriveCyclePort;
        port powerDataOut : PowerDataPort;
        port fuelEconomyOut : FuelEconomyPort;
        port complianceOut : CompliancePort;
        port resultsOut : ResultsPort;
    }

    part def ScenarioConfiguration {
        attribute scenarioName : String;
        attribute scenarioDescription : String;
        attribute targetMPG : Real;
        attribute targetL100km : Real;
        attribute driveMode : DriveMode;
        attribute isEnabled : Boolean;

        part vehicleSetup : VehicleAttributes;
        part cycle : DriveCycle;
    }

    part def VehicleFleet {
        attribute fleetName : String;
        attribute numberOfVehicles : Integer;

        part vehicle1 : VehicleDynamicsSystem;
        part vehicle2 : VehicleDynamicsSystem;
        part vehicle3 : VehicleDynamicsSystem;

        part scenario1 : ScenarioConfiguration;
        part scenario2 : ScenarioConfiguration;
        part scenario3 : ScenarioConfiguration;
    }

    part topLevelSystem : VehicleDynamicsSystem {
        attribute systemName = "VehicleDynamicsAnalysisSystem";
        attribute systemVersion = "1.0";
        attribute fuelEconomyTarget30MPG = 30.0;
        attribute targetFuelConsumptionL100km = 7.84;

        part vehicleAttributes : VehicleAttributes {
            attribute curbWeight = 1500.0;
            attribute payload = 500.0;
            attribute totalMass = 2000.0;
            attribute wheelDiameter = 0.65;
            attribute transmissionEfficiency = 0.92;
            attribute fuelEconomyUrban = 25.0;
            attribute fuelEconomyHighway = 35.0;
            attribute fuelEconomyMixed = 30.0;
            attribute dragCoefficient = 0.30;
            attribute frontalArea = 2.2;
            attribute rollingResistanceCoefficient = 0.015;
            attribute fuelType = FuelType::Gasoline;
        }

        part driveCycleUrban : DriveCycle {
            attribute cycleName = "UrbanDriveCycle";
            attribute totalDuration = 1369.0;
            attribute totalDistance = 12.07;
            attribute driveMode = DriveMode::Urban;
            attribute averageSpeed = 31.7;
            attribute maxSpeed = 91.25;
            attribute minSpeed = 0.0;

            part point1 : DriveCyclePoint {
                attribute timeStamp = 0.0;
                attribute position = 0.0;
                attribute speed = 0.0;
                attribute acceleration = 0.0;
            }
            part point2 : DriveCyclePoint {
                attribute timeStamp = 10.0;
                attribute position = 25.0;
                attribute speed = 5.0;
                attribute acceleration = 0.5;
            }
            part point3 : DriveCyclePoint {
                attribute timeStamp = 30.0;
                attribute position = 150.0;
                attribute speed = 15.0;
                attribute acceleration = 0.5;
            }
            part point4 : DriveCyclePoint {
                attribute timeStamp = 60.0;
                attribute position = 500.0;
                attribute speed = 25.0;
                attribute acceleration = 0.3;
            }
            part point5 : DriveCyclePoint {
                attribute timeStamp = 120.0;
                attribute position = 1500.0;
                attribute speed = 35.0;
                attribute acceleration = 0.0;
            }
            part point6 : DriveCyclePoint {
                attribute timeStamp = 200.0;
                attribute position = 3000.0;
                attribute speed = 20.0;
                attribute acceleration = -0.5;
            }
            part point7 : DriveCyclePoint {
                attribute timeStamp = 300.0;
                attribute position = 4500.0;
                attribute speed = 0.0;
                attribute acceleration = -1.0;
            }
            part point8 : DriveCyclePoint {
                attribute timeStamp = 400.0;
                attribute position = 6000.0;
                attribute speed = 30.0;
                attribute acceleration = 0.8;
            }
            part point9 : DriveCyclePoint {
                attribute timeStamp = 500.0;
                attribute position = 8000.0;
                attribute speed = 25.0;
                attribute acceleration = -0.2;
            }
            part point10 : DriveCyclePoint {
                attribute timeStamp = 600.0;
                attribute position = 10000.0;
                attribute speed = 0.0;
                attribute acceleration = -0.8;
            }
        }

        part driveCycleHighway : DriveCycle {
            attribute cycleName = "HighwayDriveCycle";
            attribute totalDuration = 765.0;
            attribute totalDistance = 16.51;
            attribute driveMode = DriveMode::Highway;
            attribute averageSpeed = 77.7;
            attribute maxSpeed = 96.4;
            attribute minSpeed = 0.0;

            part point1 : DriveCyclePoint {
                attribute timeStamp = 0.0;
                attribute position = 0.0;
                attribute speed = 0.0;
                attribute acceleration = 0.0;
            }
            part point2 : DriveCyclePoint {
                attribute timeStamp = 30.0;
                attribute position = 500.0;
                attribute speed = 60.0;
                attribute acceleration = 2.0;
            }
            part point3 : DriveCyclePoint {
                attribute timeStamp = 60.0;
                attribute position = 2000.0;
                attribute speed = 80.0;
                attribute acceleration = 0.67;
            }
            part point4 : DriveCyclePoint {
                attribute timeStamp = 120.0;
                attribute position = 5500.0;
                attribute speed = 90.0;
                attribute acceleration = 0.17;
            }
            part point5 : DriveCyclePoint {
                attribute timeStamp = 200.0;
                attribute position = 10000.0;
                attribute speed = 96.4;
                attribute acceleration = 0.08;
            }
            part point6 : DriveCyclePoint {
                attribute timeStamp = 300.0;
                attribute position = 14000.0;
                attribute speed = 88.0;
                attribute acceleration = -0.08;
            }
            part point7 : DriveCyclePoint {
                attribute timeStamp = 400.0;
                attribute position = 17500.0;
                attribute speed = 80.0;
                attribute acceleration = -0.08;
            }
            part point8 : DriveCyclePoint {
                attribute timeStamp = 500.0;
                attribute position = 21000.0;
                attribute speed = 70.0;
                attribute acceleration = -0.1;
            }
            part point9 : DriveCyclePoint {
                attribute timeStamp = 600.0;
                attribute position = 24000.0;
                attribute speed = 50.0;
                attribute acceleration = -0.2;
            }
            part point10 : DriveCyclePoint {
                attribute timeStamp = 765.0;
                attribute position = 26572.0;
                attribute speed = 0.0;
                attribute acceleration = -0.3;
            }
        }

        part driveCycleMixed : DriveCycle {
            attribute cycleName = "MixedDriveCycle";
            attribute totalDuration = 2134.0;
            attribute totalDistance = 28.58;
            attribute driveMode = DriveMode::Mixed;
            attribute averageSpeed = 48.3;
            attribute maxSpeed = 96.4;
            attribute minSpeed = 0.0;

            part point1 : DriveCyclePoint {
                attribute timeStamp = 0.0;
                attribute position = 0.0;
                attribute speed = 0.0;
                attribute acceleration = 0.0;
            }
            part point2 : DriveCyclePoint {
                attribute timeStamp = 100.0;
                attribute position = 1000.0;
                attribute speed = 30.0;
                attribute acceleration = 0.3;
            }
            part point3 : DriveCyclePoint {
                attribute timeStamp = 300.0;
                attribute position = 5000.0;
                attribute speed = 60.0;
                attribute acceleration = 0.15;
            }
            part point4 : DriveCyclePoint {
                attribute timeStamp = 600.0;
                attribute position = 12000.0;
                attribute speed = 80.0;
                attribute acceleration = 0.067;
            }
            part point5 : DriveCyclePoint {
                attribute timeStamp = 900.0;
                attribute position = 20000.0;
                attribute speed = 90.0;
                attribute acceleration = 0.033;
            }
            part point6 : DriveCyclePoint {
                attribute timeStamp = 1100.0;
                attribute position = 24000.0;
                attribute speed = 40.0;
                attribute acceleration = -0.25;
            }
            part point7 : DriveCyclePoint {
                attribute timeStamp = 1300.0;
                attribute position = 26000.0;
                attribute speed = 0.0;
                attribute acceleration = -0.2;
            }
            part point8 : DriveCyclePoint {
                attribute timeStamp = 1500.0;
                attribute position = 28000.0;
                attribute speed = 30.0;
                attribute acceleration = 0.15;
            }
            part point9 : DriveCyclePoint {
                attribute timeStamp = 1800.0;
                attribute position = 31000.0;
                attribute speed = 20.0;
                attribute acceleration = -0.033;
            }
            part point10 : DriveCyclePoint {
                attribute timeStamp = 2134.0;
                attribute position = 46000.0;
                attribute speed = 0.0;
                attribute acceleration = -0.06;
            }
        }

        part powerCalculator : PowerDemandCalculator {
            attribute gravitationalAcceleration = 9.81;
            attribute airDensity = 1.225;
            attribute calculatedTractivePower = 0.0;
            attribute aerodynamicDragPower = 0.0;
            attribute rollingResistancePower = 0.0;
            attribute gradePower = 0.0;
            attribute inertiaPower = 0.0;
            attribute totalPowerDemand = 0.0;
            attribute enginePowerRequired = 0.0;
        }

        part accelerationCalc : AccelerationCalculator {
            attribute currentSpeed = 0.0;
            attribute previousSpeed = 0.0;
            attribute timeInterval = 1.0;
            attribute calculatedAcceleration = 0.0;
            attribute maxAcceleration = 0.0;
            attribute minAcceleration = 0.0;
            attribute averageAcceleration = 0.0;
        }

        part speedPositionTracker : SpeedPositionTracker {
            attribute currentPosition = 0.0;
            attribute currentSpeed = 0.0;
            attribute elapsedTime = 0.0;
            attribute distanceTraveled = 0.0;
            attribute speedAtTime = 0.0;
            attribute positionAtTime = 0.0;
        }

        part fuelEconomyCalc : FuelEconomyCalculator {
            attribute totalFuelConsumed = 0.0;
            attribute totalDistanceCovered = 0.0;
            attribute fuelEconomyMPG = 30.0;
            attribute fuelEconomyL100km = 7.84;
            attribute urbanFuelConsumption = 0.0;
            attribute highwayFuelConsumption = 0.0;
            attribute mixedFuelConsumption = 0.0;
            attribute energyConsumptionRate = 0.0;
        }

        part complianceVerifier : ComplianceVerifier {
            attribute targetFuelEconomyMPG = 30.0;
            attribute targetFuelEconomyL100km = 7.84;
            attribute actualFuelEconomyMPG = 30.0;
            attribute actualFuelEconomyL100km = 7.84;
            attribute complianceStatus = ComplianceStatus::Pass;
            attribute marginMPG = 0.0;
            attribute marginL100km = 0.0;
            attribute isCompliant = true;
            attribute verificationNotes = "Vehicle meets 30 MPG fuel economy target";
        }

        part configManager : ConfigurationManager {
            attribute configurationName = "DefaultConfiguration";
            attribute isActive = true;
            attribute lastUpdated = "2024-01-01";
            attribute numberOfScenarios = 3;
            attribute selectedScenarioIndex = 1;
        }

        part analysisResult : PerformanceAnalysisResult {
            attribute analysisId = "ANALYSIS-001";
            attribute analysisTimestamp = "2024-01-01T00:00:00";
            attribute peakPowerDemand = 85.0;
            attribute averagePowerDemand = 20.0;
            attribute peakAcceleration = 2.0;
            attribute peakDeceleration = -3.0;
            attribute topSpeedAchieved = 96.4;
            attribute totalEnergyConsumed = 45.0;
            attribute overallFuelEconomyMPG = 30.0;
            attribute overallFuelEconomyL100km = 7.84;
            attribute complianceResult = ComplianceStatus::Pass;
        }

        port vehicleDataIn : VehicleDataPort;
        port driveCycleIn : DriveCyclePort;
        port powerDataOut : PowerDataPort;
        port fuelEconomyOut : FuelEconomyPort;
        port complianceOut : CompliancePort;
        port resultsOut : ResultsPort;

        connect vehicleDataIn to powerCalculator;
        connect driveCycleIn to accelerationCalc;
        connect powerCalculator to powerDataOut;
        connect fuelEconomyCalc to fuelEconomyOut;
        connect complianceVerifier to complianceOut;
        connect analysisResult to resultsOut;
    }
}