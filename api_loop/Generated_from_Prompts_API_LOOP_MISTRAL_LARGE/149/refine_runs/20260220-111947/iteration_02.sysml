package PubSubSystem {
    private import ScalarValues::*;

    enum def Topic {
        enum Weather;
        enum News;
        enum Sports;
    }

    item def Message {
        attribute topic : Topic;
        attribute content : String;
    }

    port def PublishPort {
        out item outMessage : Message;
    }

    port def SubscribePort {
        in item inMessage : Message;
        in item topicRequest : Topic;
    }

    port def DistributionPort {
        out item outMessage : Message;
    }

    part def Publisher {
        port publishPort : PublishPort;
    }

    part def Subscriber {
        port subscribePort : SubscribePort;
        attribute subscribedTopics : Topic[0..*];
    }

    part def Server {
        port publishReceivePort : PublishPort;
        port subscribeReceivePort : SubscribePort;
        port distributionPort : DistributionPort;
        attribute subscriptions : Subscriber[0..*];
    }

    part publisher1 :> Publisher;
    part subscriber1 :> Subscriber;
    part subscriber2 :> Subscriber;
    part server :> Server;

    connect publisher1.publishPort to server.publishReceivePort;
    connect server.distributionPort to subscriber1.subscribePort;
    connect server.distributionPort to subscriber2.subscribePort;
}