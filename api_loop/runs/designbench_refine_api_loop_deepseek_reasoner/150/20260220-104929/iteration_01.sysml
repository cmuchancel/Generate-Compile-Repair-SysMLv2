package PubSubSystem {
    private import ScalarValues::*;
    
    item def SubscriptionRequest {
        attribute topic : string;
    }
    
    item def Message {
        attribute topic : string;
        attribute content : string;
    }
    
    port def PublisherPort;
    
    port def SubscriberPort;
    
    port def ServerPort;
    
    part def Producer {
        port pub : PublisherPort;
        attribute id : string;
    }
    
    part def Consumer {
        port sub : SubscriberPort;
        attribute id : string;
    }
    
    part def Server {
        port serverPort : ServerPort;
        attribute subscriptions : string[*];
    }
    
    part def System {
        part producers : Producer[*];
        part server : Server;
        part consumers : Consumer[*];
        
        connect producers.pub to server.serverPort;
        connect consumers.sub to server.serverPort;
    }
}