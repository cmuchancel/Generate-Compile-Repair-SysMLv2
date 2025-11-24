package 'TorqueGenerationSystem' {
	private import ScalarValues::*;
	package LogicalLayer {
		action def 'Generate Torque';
		action torqueFunction : 'Generate Torque';
		part def TorqueGenerator;
		part torqueGenerator : TorqueGenerator {
			perform torqueFunction;
		}
	}
	package PhysicalLayer {
		private import LogicalLayer::*;
		part def PowertrainComponent;
		part def Engine :> PowertrainComponent;
		part powertrain : PowertrainComponent;
		part engine : Engine {
			attribute generatedTorque : Real;
			perform torqueFunction;
		}
		constraint def PositiveTorque {
			in torque : Real;
			assert constraint { torque > 0 }
		}
		part torqueCheck : PositiveTorque {
			in torque = engine.generatedTorque;
		}
		allocation def LogicalToPhysical {
			end logical : TorqueGenerator;
			end physical : Engine;
		}
		allocation torqueAllocation : LogicalToPhysical allocate torqueGenerator to engine;
	}
}
