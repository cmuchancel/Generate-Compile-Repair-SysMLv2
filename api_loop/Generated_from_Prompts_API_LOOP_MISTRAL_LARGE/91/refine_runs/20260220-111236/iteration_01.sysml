package BatteryRiskAnalysis {
    private import ScalarValues::*;

    enum def SeverityLevel {
        enum low;
        enum medium;
        enum high;
        enum critical;
    }

    part def Battery {
        attribute level : Real;
        attribute agingFactor : Real = 0.0;
        attribute minThreshold : Real = 20.0;
        attribute isLow : Boolean = false;
        attribute failureProbability : Real = 0.0;
    }

    part def Device {
        part battery : Battery;
        attribute isShutdown : Boolean = false;
    }

    part def RiskAnalyzer {
        attribute detectedRisk : String;
        attribute severity : SeverityLevel;
    }

    part system : Device {
        part analyzer : RiskAnalyzer;
    }

    connect system.battery to system.analyzer;
}