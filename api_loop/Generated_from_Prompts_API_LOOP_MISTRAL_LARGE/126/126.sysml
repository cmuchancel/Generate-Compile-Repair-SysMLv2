package LayeredSystem {
    private import ScalarValues::*;

    part def ApplicationLayer;
    part def ServiceLayer;
    part def DataLayer;

    attribute x : Real;
    attribute y : Real;
    attribute z : Real = x + y;

    part application : ApplicationLayer;
    part service : ServiceLayer {
        port servicePort : ServicePort;
    }
    part data : DataLayer {
        port dataPort : DataPort;
    }

    port def ServicePort;
    port def DataPort;

    connect application to service::servicePort;
    connect service::servicePort to data::dataPort;
}