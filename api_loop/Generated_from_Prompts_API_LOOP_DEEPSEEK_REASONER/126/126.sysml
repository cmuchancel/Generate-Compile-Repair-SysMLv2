package SystemModel {
    private import ScalarValues::*;

    part def ApplicationLayer {
        port serviceReq : ServiceLayerPort;
        attribute z : Real;
    }

    part def ServiceLayer {
        port provided : ServiceLayerPort;
        port dataReq : DataLayerPort;
        attribute x : Real;
        attribute y : Real;
    }

    part def DataLayer {
        port provided : DataLayerPort;
        attribute storedX : Real;
        attribute storedY : Real;
    }

    port def ServiceLayerPort;

    port def DataLayerPort;

    part System {
        part appLayer : ApplicationLayer;
        part svcLayer : ServiceLayer;
        part dataLayer : DataLayer;

        connect appLayer.serviceReq to svcLayer.provided;
        connect svcLayer.dataReq to dataLayer.provided;
    }
}