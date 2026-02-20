package PubSubSystem {
    private import ScalarValues::*;

    item def SubscriptionRequest {
        attribute topic : String;
    }

    item def Message {
        attribute topic : String;
        attribute content : String;
    }

    port def PublisherPort;

    port def SubscriberPort;

    port def ServerPort;

    part def Producer {
        port pub : PublisherPort;
        attribute id : String;
    }

    part def Consumer {
        port sub : SubscriberPort;
        attribute id : String;
    }

    part def Server {
        port serverPort : ServerPort;
        attribute subscriptions : String[*];
    }

    part def System {
        part producers : Producer[*];
        part server : Server;
        part consumers : Consumer[*];

        connect producers.pub to server.serverPort;
        connect consumers.sub to server.serverPort;
    }
}