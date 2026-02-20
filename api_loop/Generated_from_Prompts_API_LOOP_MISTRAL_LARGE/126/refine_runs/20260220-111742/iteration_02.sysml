package LayeredSystem {
    private import ScalarValues::*;

    part def ApplicationLayer;
    part def ServiceLayer;
    part def DataLayer;

    attribute x : Real;
    attribute y : Real;
    attribute z : Real = x + y;

    part application : ApplicationLayer;
    part service : ServiceLayer;
    part data : DataLayer;

    port def ServicePort;
    port def DataPort;

    port servicePort : ServicePort in service;
    port dataPort : DataPort in data;

    connect application to service.servicePort;
    connect service.servicePort to data.dataPort;
}