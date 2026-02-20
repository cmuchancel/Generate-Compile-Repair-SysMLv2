package PublishSubscribeSystem {
    private import ScalarValues::*;

    enum def Topic {
        enum Weather;
        enum News;
        enum Sports;
    }

    part def Producer {
        port publishPort : PublishPort;
    }

    part def Consumer {
        port subscribePort : SubscribePort;
        port receivePort : ReceivePort;
    }

    part def Server {
        port publishReceivePort : PublishPort;
        port subscribeHandlePort : SubscribePort;
        port pushPort : PushPort;

        attribute subscriptions : String = "";
    }

    port def PublishPort {
        attribute topic : Topic;
        attribute msg : String;
    }

    port def SubscribePort {
        attribute topic : Topic;
    }

    port def ReceivePort {
        attribute msg : String;
    }

    port def PushPort {
        attribute topic : Topic;
        attribute msg : String;
    }

    item def Message {
        attribute topic : Topic;
        attribute content : String;
    }

    part producer1 : Producer;
    part producer2 : Producer;
    part consumer1 : Consumer;
    part consumer2 : Consumer;
    part server : Server;

    connect producer1.publishPort to server.publishReceivePort;
    connect producer2.publishPort to server.publishReceivePort;
    connect consumer1.subscribePort to server.subscribeHandlePort;
    connect consumer2.subscribePort to server.subscribeHandlePort;
    connect server.pushPort to consumer1.receivePort;
    connect server.pushPort to consumer2.receivePort;
}