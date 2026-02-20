package LayeredSystem {
    private import ScalarValues::*;

    attribute def XAttribute { attribute value : Real; }
    attribute def YAttribute { attribute value : Real; }
    attribute def ZAttribute { attribute value : Real; }

    part def DataLayer {
        attribute dataStore : String;
        port dataPort : DataPort;
    }

    part def ServiceLayer {
        attribute serviceState : String;
        port dataConsumerPort : DataPort;
        port servicePort : ServicePort;
    }

    part def ApplicationLayer {
        attribute appState : String;
        port serviceConsumerPort : ServicePort;
    }

    port def DataPort {
        attribute transferId : String;
    }

    port def ServicePort {
        attribute requestId : String;
    }

    part def LayeredSystemRoot {
        attribute x : Real;
        attribute y : Real;
        attribute z : Real;

        part dataLayer : DataLayer;
        part serviceLayer : ServiceLayer;
        part applicationLayer : ApplicationLayer;

        connect dataLayer.dataPort to serviceLayer.dataConsumerPort;
        connect serviceLayer.servicePort to applicationLayer.serviceConsumerPort;
    }
}