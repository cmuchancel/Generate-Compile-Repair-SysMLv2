package 'EngineTradeoffAnalysis' {
	private import ScalarValues::*;
	part def Cylinder;
	part def Piston;
	part def ConnectingRod;
	part def Crankshaft;
	part def Engine {
		attribute power : Real;
		attribute mass : Real;
		attribute efficiency : Real;
		attribute reliability : Real;
		attribute cost : Real;
		part cylinders : Cylinder[*];
		part pistons : Piston[*];
		part connectingRods : ConnectingRod[*];
		part crankshaft : Crankshaft;
	}
	part def FourCylinderEngine :> Engine {
		part cylinders : Cylinder[4];
		part pistons : Piston[4];
		part connectingRods : ConnectingRod[4];
	}
	part def SixCylinderEngine :> Engine {
		part cylinders : Cylinder[6];
		part pistons : Piston[6];
		part connectingRods : ConnectingRod[6];
	}
	calc def TradeoffScore {
		in power : Real;
		in mass : Real;
		in efficiency : Real;
		in reliability : Real;
		in cost : Real;
		return score : Real;
	}
	requirement def 'Engine Selection Requirement' {
		text = "Select the engine that best balances power, mass, efficiency, reliability, and cost.";
	}
	requirement def 'Selection Explanation Requirement' {
		text = "Provide a rationale for the recommended engine configuration.";
	}
	analysis def 'Engine Tradeoff Analysis';
	part tradeoffSystem {
		part candidateEngineA : FourCylinderEngine {
			attribute power = 250;
			attribute mass = 220;
			attribute efficiency = 0.32;
			attribute reliability = 0.95;
			attribute cost = 18;
		}
		part candidateEngineB : SixCylinderEngine {
			attribute power = 350;
			attribute mass = 260;
			attribute efficiency = 0.30;
			attribute reliability = 0.93;
			attribute cost = 24;
		}
		calc scoreA : TradeoffScore {
			in power = candidateEngineA.power;
			in mass = candidateEngineA.mass;
			in efficiency = candidateEngineA.efficiency;
			in reliability = candidateEngineA.reliability;
			in cost = candidateEngineA.cost;
		}
		calc scoreB : TradeoffScore {
			in power = candidateEngineB.power;
			in mass = candidateEngineB.mass;
			in efficiency = candidateEngineB.efficiency;
			in reliability = candidateEngineB.reliability;
			in cost = candidateEngineB.cost;
		}
		requirement selectionRequirement : 'Engine Selection Requirement';
		requirement explanationRequirement : 'Selection Explanation Requirement';
		analysis analysisCase : 'Engine Tradeoff Analysis';
	}
}
