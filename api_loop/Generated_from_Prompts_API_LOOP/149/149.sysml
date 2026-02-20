package PubSubSystem {
  private import ScalarValues::*;

  item def Message {
    attribute topic : String;
    attribute content : String;
  }

  item def Topic {
    attribute name : String;
  }

  item def Subscription {
    attribute topic : String;
  }

  port def PublishPort;
  port def SubscribePort;
  port def DeliverPort;

  part def Publisher {
    attribute publishTopic : String;
    port publish : PublishPort;
  }

  part def Subscriber {
    attribute subscribedTopic : String;
    port subscribe : SubscribePort;
    port receive : DeliverPort;
  }

  part def Server {
    attribute subscriptionTable : String;
    port publishIn : PublishPort;
    port subscribeIn : SubscribePort;
    port deliverOut : DeliverPort;
  }

  part def PubSubSystem {
    part server : Server;
    part producer1 : Publisher;
    part consumer1 : Subscriber;

    connect producer1.publish to server.publishIn;
    connect consumer1.subscribe to server.subscribeIn;
    connect server.deliverOut to consumer1.receive;
  }
}