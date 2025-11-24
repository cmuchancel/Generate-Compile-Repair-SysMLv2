package 'SystemAssemblyIntegration' {
	private import ScalarValues::*;
	action def SoftwareFunction;
	part def SoftwareDesign {
		attribute messageFormat : String;
		attribute dataFormat : String;
		action functions : SoftwareFunction[*];
	}
	part def ComputerSubsystem {
		action integratedFunctions : SoftwareFunction[*];
	}
	part def StorageSubsystem {
		attribute accessibleMessageFormat : String;
		attribute accessibleDataFormat : String;
	}
	part def SystemAssembly {
		part computer : ComputerSubsystem;
		part storage : StorageSubsystem;
		part softwareDesign : SoftwareDesign;
		bind computer.integratedFunctions = softwareDesign.functions;
		bind storage.accessibleMessageFormat = softwareDesign.messageFormat;
		bind storage.accessibleDataFormat = softwareDesign.dataFormat;
	}
}
