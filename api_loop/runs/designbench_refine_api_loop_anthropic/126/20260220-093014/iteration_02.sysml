package LayeredArchitectureSystem {
    private import ScalarValues::*;

    enum def LayerType {
        enum Application;
        enum Service;
        enum Data;
    }

    part def DataLayer {
        attribute layerName : String = "DataLayer";
        attribute x : Real;
        attribute y : Real;
    }

    part def ServiceLayer {
        attribute layerName : String = "ServiceLayer";
        attribute x : Real;
        attribute y : Real;
        attribute z : Real;
        port dataPort : DataAccessPort;
    }

    part def ApplicationLayer {
        attribute layerName : String = "ApplicationLayer";
        port servicePort : ServiceAccessPort;
    }

    port def DataAccessPort {
        attribute connected : Boolean;
    }

    port def ServiceAccessPort {
        attribute connected : Boolean;
    }

    part system {
        attribute x : Real;
        attribute y : Real;
        attribute z : Real;

        part appLayer : ApplicationLayer {
            port servicePort : ServiceAccessPort;
        }

        part svcLayer : ServiceLayer {
            attribute x : Real;
            attribute y : Real;
            attribute z : Real;
            port dataPort : DataAccessPort;
            port servicePort : ServiceAccessPort;
        }

        part dataLayer : DataLayer {
            attribute x : Real;
            attribute y : Real;
        }

        connect appLayer.servicePort to svcLayer.servicePort;
        connect svcLayer.dataPort to dataLayer;
    }
}