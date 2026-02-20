package LayeredSystem {
  private import ScalarValues::*;

  port def AppServicePort;
  port def ServiceDataPort;

  part def ApplicationLayer {
    port appToService : AppServicePort;
  }

  part def ServiceLayer {
    port serviceFromApp : AppServicePort;
    port serviceToData : ServiceDataPort;
  }

  part def DataLayer {
    port dataFromService : ServiceDataPort;
  }

  part def System {
    attribute x : Real;
    attribute y : Real;
    attribute z : Real = x + y;

    part appLayer : ApplicationLayer;
    part serviceLayer : ServiceLayer;
    part dataLayer : DataLayer;

    connect appLayer.appToService to serviceLayer.serviceFromApp;
    connect serviceLayer.serviceToData to dataLayer.dataFromService;
  }

  part system : System;
}