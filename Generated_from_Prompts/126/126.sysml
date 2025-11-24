package 'LayeredSystemArchitecture' {
	private import ScalarValues::*;
	port def ServicePort;
	port def DataPort;
	connection def ServiceInvocation {
		end application : ServicePort;
		end service : ServicePort;
	}
	connection def DataAccessLink {
		end service : DataPort;
		end data : DataPort;
	}
	part def ApplicationLayer {
		port serviceRequest : ServicePort;
	}
	part def ServiceLayer {
		port applicationInterface : ServicePort;
		port dataInterface : DataPort;
	}
	part def DataLayer {
		port storageInterface : DataPort;
	}
	part def LayeredSystem {
		attribute x : Real;
		attribute y : Real;
		attribute z : Real;
		part applicationLayer : ApplicationLayer;
		part serviceLayer : ServiceLayer;
		part dataLayer : DataLayer;
		connection appToService : ServiceInvocation {
			end application = applicationLayer.serviceRequest;
			end service = serviceLayer.applicationInterface;
		}
		connection serviceToData : DataAccessLink {
			end service = serviceLayer.dataInterface;
			end data = dataLayer.storageInterface;
		}
	}
	constraint def ZDependency {
		in x : Real;
		in y : Real;
		in z : Real;
		assert constraint { z = x + y; }
	}
	part architectureInstance : LayeredSystem {
		attribute x = 5;
		attribute y = 7;
		part zConstraint : ZDependency {
			in x = architectureInstance::x;
			in y = architectureInstance::y;
			in z = architectureInstance::z;
		}
	}
}
