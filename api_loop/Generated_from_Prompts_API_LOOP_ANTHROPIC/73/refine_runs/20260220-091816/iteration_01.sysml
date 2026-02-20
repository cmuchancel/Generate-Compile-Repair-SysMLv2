package VehicleFuelEconomyAnalysis {
    private import ScalarValues::*;

    enum def DrivingScenario {
        enum Urban;
        enum Highway;
    }

    enum def ComplianceStatus {
        enum Pass;
        enum Fail;
    }

    part def VehicleParameters {
        attribute curbWeight : Real;
        attribute load : Real;
        attribute wheelDiameter : Real;
        attribute transmissionEfficiency : Real;
        attribute dragCoefficient : Real;
        attribute frontalArea : Real;
        attribute rollingResistanceCoefficient : Real;
    }

    part def DrivingScenarioConfig {
        attribute scenario : DrivingScenario;
        attribute averageSpeed : Real;
        attribute distanceKm : Real;
        attribute peakAcceleration : Real;
        attribute idleTimeFraction : Real;
    }

    part def WheelPowerAnalysis {
        attribute requiredWheelPower : Real;
        attribute aerodynamicDragForce : Real;
        attribute rollingResistanceForce : Real;
        attribute accelerationForce : Real;
        attribute totalTractiveForce : Real;
    }

    part def VehicleDynamicsCalculator {
        attribute totalMass : Real;
        attribute effectiveWheelRadius : Real;
        attribute linearAcceleration : Real;
        attribute kineticEnergyChange : Real;
        attribute gradeResistanceForce : Real;
    }

    part def EnginePowerEstimator {
        attribute requiredEnginePower : Real;
        attribute engineThermalEfficiency : Real;
        attribute fuelEnergyDensity : Real;
        attribute fuelFlowRate : Real;
        attribute idleFuelConsumption : Real;
    }

    part def FuelEconomyResult {
        attribute fuelConsumptionLitersPer100Km : Real;
        attribute fuelEconomyMilesPerGallon : Real;
        attribute requirementThresholdMpg : Real;
        attribute complianceStatus : ComplianceStatus;
    }

    part def FuelEconomySimulation {
        attribute airDensity : Real;
        attribute gravityAcceleration : Real;

        part vehicleParams : VehicleParameters;
        part urbanScenario : DrivingScenarioConfig;
        part highwayScenario : DrivingScenarioConfig;
        part wheelPowerUrban : WheelPowerAnalysis;
        part wheelPowerHighway : WheelPowerAnalysis;
        part dynamicsCalculator : VehicleDynamicsCalculator;
        part engineEstimator : EnginePowerEstimator;
        part urbanResult : FuelEconomyResult;
        part highwayResult : FuelEconomyResult;
    }

    part def FuelEconomySystem {
        part simulation : FuelEconomySimulation;
        part complianceUrban : FuelEconomyResult;
        part complianceHighway : FuelEconomyResult;
    }

    part fuelEconomySystem : FuelEconomySystem {
        part simulation : FuelEconomySimulation {
            attribute airDensity : Real = 1.225;
            attribute gravityAcceleration : Real = 9.81;

            part vehicleParams : VehicleParameters {
                attribute curbWeight : Real = 1500.0;
                attribute load : Real = 150.0;
                attribute wheelDiameter : Real = 0.65;
                attribute transmissionEfficiency : Real = 0.92;
                attribute dragCoefficient : Real = 0.30;
                attribute frontalArea : Real = 2.2;
                attribute rollingResistanceCoefficient : Real = 0.012;
            }

            part urbanScenario : DrivingScenarioConfig {
                attribute scenario : DrivingScenario = DrivingScenario::Urban;
                attribute averageSpeed : Real = 35.0;
                attribute distanceKm : Real = 100.0;
                attribute peakAcceleration : Real = 1.5;
                attribute idleTimeFraction : Real = 0.20;
            }

            part highwayScenario : DrivingScenarioConfig {
                attribute scenario : DrivingScenario = DrivingScenario::Highway;
                attribute averageSpeed : Real = 110.0;
                attribute distanceKm : Real = 100.0;
                attribute peakAcceleration : Real = 0.5;
                attribute idleTimeFraction : Real = 0.02;
            }

            part wheelPowerUrban : WheelPowerAnalysis {
                attribute aerodynamicDragForce : Real = 68.4;
                attribute rollingResistanceForce : Real = 193.6;
                attribute accelerationForce : Real = 2475.0;
                attribute totalTractiveForce : Real = 2737.0;
                attribute requiredWheelPower : Real = 26.6;
            }

            part wheelPowerHighway : WheelPowerAnalysis {
                attribute aerodynamicDragForce : Real = 676.5;
                attribute rollingResistanceForce : Real = 193.6;
                attribute accelerationForce : Real = 825.0;
                attribute totalTractiveForce : Real = 1695.1;
                attribute requiredWheelPower : Real = 51.8;
            }

            part dynamicsCalculator : VehicleDynamicsCalculator {
                attribute totalMass : Real = 1650.0;
                attribute effectiveWheelRadius : Real = 0.325;
                attribute linearAcceleration : Real = 1.5;
                attribute kineticEnergyChange : Real = 1856.25;
                attribute gradeResistanceForce : Real = 0.0;
            }

            part engineEstimator : EnginePowerEstimator {
                attribute engineThermalEfficiency : Real = 0.35;
                attribute fuelEnergyDensity : Real = 34.2;
                attribute requiredEnginePower : Real = 28.9;
                attribute fuelFlowRate : Real = 2.35;
                attribute idleFuelConsumption : Real = 0.5;
            }

            part urbanResult : FuelEconomyResult {
                attribute fuelConsumptionLitersPer100Km : Real = 9.8;
                attribute fuelEconomyMilesPerGallon : Real = 24.0;
                attribute requirementThresholdMpg : Real = 30.0;
                attribute complianceStatus : ComplianceStatus = ComplianceStatus::Fail;
            }

            part highwayResult : FuelEconomyResult {
                attribute fuelConsumptionLitersPer100Km : Real = 6.8;
                attribute fuelEconomyMilesPerGallon : Real = 34.6;
                attribute requirementThresholdMpg : Real = 30.0;
                attribute complianceStatus : ComplianceStatus = ComplianceStatus::Pass;
            }
        }

        part complianceUrban : FuelEconomyResult {
            attribute fuelConsumptionLitersPer100Km : Real = 9.8;
            attribute fuelEconomyMilesPerGallon : Real = 24.0;
            attribute requirementThresholdMpg : Real = 30.0;
            attribute complianceStatus : ComplianceStatus = ComplianceStatus::Fail;
        }

        part complianceHighway : FuelEconomyResult {
            attribute fuelConsumptionLitersPer100Km : Real = 6.8;
            attribute fuelEconomyMilesPerGallon : Real = 34.6;
            attribute requirementThresholdMpg : Real = 30.0;
            attribute complianceStatus : ComplianceStatus = ComplianceStatus::Pass;
        }
    }
}