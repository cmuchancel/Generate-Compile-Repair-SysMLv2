package PubSubModel {
  private import ScalarValues::*;
  enum def TopicEnum {
    enum TopicA;
    enum TopicB;
    enum TopicC;
  }
  item def Message {
    attribute topic : TopicEnum;
    attribute payload : String;
  }
  item def Subscription {
    attribute topic : TopicEnum;
    attribute consumerId : String;
  }
  port def PublishPort {
  }
  port def SubscribePort {
  }
  port def DeliverPort {
  }
  part def Producer {
    port publish : PublishPort;
  }
  part def Consumer {
    port subscribe : SubscribePort;
    port receive : DeliverPort;
  }
  part def Server {
    port fromProducers : PublishPort;
    port fromConsumers : SubscribePort;
    port toConsumers : DeliverPort;
  }
  part PubSubSystem {
    part producer : Producer;
    part consumer : Consumer;
    part server : Server;
    connect producer.publish to server.fromProducers;
    connect consumer.subscribe to server.fromConsumers;
    connect server.toConsumers to consumer.receive;
  }
}