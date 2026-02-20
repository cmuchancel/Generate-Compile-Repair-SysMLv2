package LayeredSystem {
    private import ScalarValues::*;

    enum def LayerRole {
        enum APPLICATION;
        enum SERVICE;
        enum DATA;
    }

    port def DataLayerPort {
        attribute connected : Boolean;
    }

    port def ServiceLayerPort {
        attribute connected : Boolean;
    }

    part def DataLayer {
        attribute storageCapacity : Real;
        port dataPort : DataLayerPort;
    }

    part def ServiceLayer {
        attribute serviceCount : Integer;
        port dataPort : DataLayerPort;
        port servicePort : ServiceLayerPort;
    }

    part def ApplicationLayer {
        attribute userCount : Integer;
        port servicePort : ServiceLayerPort;
    }

    part def System {
        attribute x : Real;
        attribute y : Real;
        attribute z : Real;

        part appLayer : ApplicationLayer;
        part svcLayer : ServiceLayer;
        part dataLayer : DataLayer;

        connect appLayer.servicePort to svcLayer.servicePort;
        connect svcLayer.dataPort to dataLayer.dataPort;
    }
}