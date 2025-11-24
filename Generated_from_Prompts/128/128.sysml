package 'PowertrainFunctionAllocation' {
	package LogicalModel {
		action def ProvidePower;
		action def GenerateTorque;
		action providePower : ProvidePower {
			action generateTorque : GenerateTorque;
		}
		part def TorqueGenerator;
		part torqueGenerator : TorqueGenerator {
			perform providePower.generateTorque;
		}
	}
	package PhysicalModel {
		private import LogicalModel::*;
		part def Powertrain;
		part powertrain : Powertrain {
			part engine {
				perform providePower.generateTorque;
			}
		}
		allocation def FunctionalAllocation {
			end logical : TorqueGenerator;
			end physical : Powertrain;
		}
		allocation torqueGeneratorAllocation : FunctionalAllocation allocate torqueGenerator to powertrain;
	}
}
