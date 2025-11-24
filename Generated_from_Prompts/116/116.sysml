package 'ModularVehiclePlatform' {
	private import ScalarValues::*;
port def IntegrationPort;
connection def SubsystemLink {
	end fromPort : IntegrationPort;
	end toPort : IntegrationPort;
}
	part def Part1;
	part def Part2;
	part def Part3 {
		port p1 : IntegrationPort;
	}
	part def Part4;
	action def Part5BehaviorVariantA;
	action def Part5BehaviorVariantB;
	part def Part5 {
		port p2 : IntegrationPort;
		action behaviorVariantA : Part5BehaviorVariantA;
		action behaviorVariantB : Part5BehaviorVariantB;
	}
	part def Part6;
	part def SubsystemA {
		port interfacePort : IntegrationPort;
	}
	part def SubsystemB {
		port interfacePort : IntegrationPort;
	}
	part def SubsystemAOption1 :> SubsystemA {
		part part1 : Part1;
		part part2 : Part2;
		part part3 : Part3;
		port interfacePort redefines interfacePort = part3.p1;
	}
	part def SubsystemAOption2 :> SubsystemA {
		part part2 : Part2;
		part part4 : Part4;
		port interfacePort redefines interfacePort;
	}
	part def SubsystemBOption1 :> SubsystemB {
		part part5 : Part5;
		part part4 : Part4;
		port interfacePort redefines interfacePort = part5.p2;
	}
	part def SubsystemBOption2 :> SubsystemB {
		part part5 : Part5;
		part part6 : Part6;
		port interfacePort redefines interfacePort = part5.p2;
	}
part def VehicleConfig {
		part subsystemA : SubsystemA;
		part subsystemB : SubsystemB;
		connection integrationLink : SubsystemLink {
			end fromPort = subsystemA.interfacePort;
			end toPort = subsystemB.interfacePort;
		}
	}
	part VehicleConfigA : VehicleConfig {
		part subsystemA : SubsystemAOption1 redefines subsystemA;
		part subsystemB : SubsystemBOption1 redefines subsystemB;
	}
	part VehicleConfigB : VehicleConfig {
		part subsystemA : SubsystemAOption2 redefines subsystemA;
		part subsystemB : SubsystemBOption2 redefines subsystemB;
	}
}
