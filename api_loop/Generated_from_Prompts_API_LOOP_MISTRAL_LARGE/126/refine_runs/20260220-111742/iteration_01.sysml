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

    part servicePort : ServicePort in service;
    part dataPort : DataPort in data;

    connect application to servicePort;
    connect servicePort to dataPort;
}