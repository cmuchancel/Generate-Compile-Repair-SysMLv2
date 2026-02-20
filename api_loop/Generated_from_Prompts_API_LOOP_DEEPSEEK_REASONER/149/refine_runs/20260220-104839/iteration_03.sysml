package PubSubSystem {
    private import ScalarValues::*;

    enum def MessagePriority {
        enum LOW;
        enum MEDIUM;
        enum HIGH;
    }

    item def Message {
        attribute content : String;
        attribute topic : String;
        attribute priority : MessagePriority;
        attribute timestamp : String;
    }

    port def PublisherPort;
    port def SubscriberPort;
    port def BrokerPort;

    part def Publisher {
        port output : PublisherPort;
        attribute id : String;
    }

    part def Subscriber {
        port input : SubscriberPort;
        attribute id : String;
    }

    part def MessageBroker {
        port broker : BrokerPort;
        part publisherInterface : PublisherPort;
        part subscriberInterface : SubscriberPort;
        connect publisherInterface to broker;
        connect subscriberInterface to broker;
    }

    part def PubSubNetwork {
        part broker : MessageBroker;
        part publisher1 : Publisher;
        part publisher2 : Publisher;
        part subscriber1 : Subscriber;
        part subscriber2 : Subscriber;
        part subscriber3 : Subscriber;
        connect publisher1.output to broker.publisherInterface;
        connect publisher2.output to broker.publisherInterface;
        connect subscriber1.input to broker.subscriberInterface;
        connect subscriber2.input to broker.subscriberInterface;
        connect subscriber3.input to broker.subscriberInterface;
    }
}