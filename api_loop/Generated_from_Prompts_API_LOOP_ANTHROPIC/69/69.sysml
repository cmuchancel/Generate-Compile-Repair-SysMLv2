package VehicleSystem {
    private import ScalarValues::*;

    enum def FuelState {
        enum FullTank;
        enum EmptyTank;
    }

    item def Fuel;

    port def ClutchPort;
    port def TransmissionPort;

    part def Engine {
        port clutchOut : ClutchPort;
        attribute generatedTorque : Real;
        attribute ratedSpeed : Real;
    }

    part def Clutch {
        port clutchIn : ClutchPort;
        port transmissionOut : TransmissionPort;
        attribute transmittedTorque : Real;
    }

    part def Transmission {
        port transmissionIn : TransmissionPort;
        attribute inputTorque : Real;
    }

    part def FuelTank {
        attribute fuelMass : Real;
        attribute maxFuelMass : Real = 500.0;
    }

    part def Vehicle {
        attribute dryMass : Real = 1200.0;
        attribute fuelMass : Real;
        attribute totalMass : Real;
        attribute maxMassFullTank : Real = 2000.0;
        attribute maxMassEmptyTank : Real = 1500.0;

        part engine : Engine {
            attribute generatedTorque : Real = 300.0;
            attribute ratedSpeed : Real = 3000.0;
            port clutchOut : ClutchPort;
        }

        part clutch : Clutch {
            attribute transmittedTorque : Real = 300.0;
            port clutchIn : ClutchPort;
            port transmissionOut : TransmissionPort;
        }

        part transmission : Transmission {
            attribute inputTorque : Real = 300.0;
            port transmissionIn : TransmissionPort;
        }

        part fuelTank : FuelTank {
            attribute fuelMass : Real = 400.0;
            attribute maxFuelMass : Real = 500.0;
        }

        connect engine.clutchOut to clutch.clutchIn;
        connect clutch.transmissionOut to transmission.transmissionIn;
    }

    part def SpeedTorquePoint {
        attribute speedRpm : Real;
        attribute torqueNm : Real;
    }

    part speedTorqueCurve1 : SpeedTorquePoint {
        attribute speedRpm : Real = 1000.0;
        attribute torqueNm : Real = 250.0;
    }

    part speedTorqueCurve2 : SpeedTorquePoint {
        attribute speedRpm : Real = 2000.0;
        attribute torqueNm : Real = 300.0;
    }

    part speedTorqueCurve3 : SpeedTorquePoint {
        attribute speedRpm : Real = 3000.0;
        attribute torqueNm : Real = 280.0;
    }

    part speedTorqueCurve4 : SpeedTorquePoint {
        attribute speedRpm : Real = 4000.0;
        attribute torqueNm : Real = 240.0;
    }

    part speedTorqueCurve5 : SpeedTorquePoint {
        attribute speedRpm : Real = 5000.0;
        attribute torqueNm : Real = 200.0;
    }

    part vehicleFullTank : Vehicle {
        attribute dryMass : Real = 1200.0;
        attribute fuelMass : Real = 500.0;
        attribute totalMass : Real = 1700.0;
        attribute maxMassFullTank : Real = 2000.0;
        attribute maxMassEmptyTank : Real = 1500.0;
    }

    part vehicleEmptyTank : Vehicle {
        attribute dryMass : Real = 1200.0;
        attribute fuelMass : Real = 0.0;
        attribute totalMass : Real = 1200.0;
        attribute maxMassFullTank : Real = 2000.0;
        attribute maxMassEmptyTank : Real = 1500.0;
    }
}