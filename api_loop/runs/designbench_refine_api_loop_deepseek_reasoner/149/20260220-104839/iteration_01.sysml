package PubSubSystem {
    private import ScalarValues::*;
    
    enum def MessagePriority {
        LOW,
        MEDIUM,
        HIGH
    }
    
    item def Message {
        attribute content : string;
        attribute topic : string;
        attribute priority : MessagePriority;
        attribute timestamp : dateTime;
    }
    
    port def PublisherPort {
        attribute topic : string;
        attribute message : Message;
    }
    
    port def SubscriberPort {
        attribute subscribedTopics : string[*];
        attribute receivedMessages : Message[*];
    }
    
    port def BrokerPort {
        attribute allTopics : string[*];
        attribute pendingMessages : Message[*];
    }
    
    part def Publisher {
        port output : PublisherPort;
        attribute id : string;
    }
    
    part def Subscriber {
        port input : SubscriberPort;
        attribute id : string;
    }
    
    part def MessageBroker {
        port broker : BrokerPort;
        attribute subscriptions : string[*];
        attribute messageQueue : Message[*];
        
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