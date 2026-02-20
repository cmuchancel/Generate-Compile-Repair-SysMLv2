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
        attribute serviceLogic : String;
        port serviceInPort : DataPort;
        port serviceOutPort : ServicePort;
    }

    part def ApplicationLayer {
        attribute appLogic : String;
        port appInPort : ServicePort;
    }

    port def DataPort {
        attribute transferValue : Real;
    }

    port def ServicePort {
        attribute serviceValue : Real;
    }

    part def LayeredSystemRoot {
        attribute x : Real;
        attribute y : Real;
        attribute z : Real;

        part dataLayer : DataLayer;
        part serviceLayer : ServiceLayer;
        part applicationLayer : ApplicationLayer;

        connect dataLayer.dataPort to serviceLayer.serviceInPort;
        connect serviceLayer.serviceOutPort to applicationLayer.appInPort;
    }
}