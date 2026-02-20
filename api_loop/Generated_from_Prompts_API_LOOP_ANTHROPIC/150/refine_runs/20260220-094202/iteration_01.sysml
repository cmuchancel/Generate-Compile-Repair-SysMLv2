package PublishSubscribeSystem {
    private import ScalarValues::*;

    enum def MessageStatus {
        enum PENDING;
        enum DELIVERED;
        enum FAILED;
    }

    enum def SubscriptionStatus {
        enum ACTIVE;
        enum INACTIVE;
    }

    item def Topic {
        attribute topicName : String;
        attribute topicId : String;
    }

    item def Message {
        attribute content : String;
        attribute messageId : String;
        attribute status : MessageStatus;
    }

    item def SubscriptionRequest {
        attribute subscriberId : String;
        attribute topicName : String;
        attribute status : SubscriptionStatus;
    }

    item def PublishRequest {
        attribute producerId : String;
        attribute topicName : String;
        attribute messageContent : String;
    }

    item def DeliveryNotification {
        attribute consumerId : String;
        attribute messageId : String;
        attribute delivered : Boolean;
    }

    port def ProducerOutputPort {
        out item publishRequest : PublishRequest;
    }

    port def ServerProducerInputPort {
        in item publishRequest : PublishRequest;
    }

    port def ConsumerSubscribePort {
        out item subscriptionRequest : SubscriptionRequest;
        in item deliveryNotification : DeliveryNotification;
    }

    port def ServerConsumerPort {
        in item subscriptionRequest : SubscriptionRequest;
        out item deliveryNotification : DeliveryNotification;
    }

    part def Producer {
        attribute producerId : String;
        attribute producerName : String;
        port publishPort : ProducerOutputPort;
    }

    part def Consumer {
        attribute consumerId : String;
        attribute consumerName : String;
        attribute subscribedTopic : String;
        port subscribePort : ConsumerSubscribePort;
    }

    part def Server {
        attribute serverId : String;
        attribute serverName : String;
        attribute maxTopics : Integer;
        port serverInputPort : ServerProducerInputPort;
        port serverConsumerPort1 : ServerConsumerPort;
        port serverConsumerPort2 : ServerConsumerPort;
        port serverConsumerPort3 : ServerConsumerPort;
    }

    part def PublishSubscribeSystem {
        part producer1 : Producer {
            attribute producerId = "P001";
            attribute producerName = "ProducerOne";
        }

        part producer2 : Producer {
            attribute producerId = "P002";
            attribute producerName = "ProducerTwo";
        }

        part consumer1 : Consumer {
            attribute consumerId = "C001";
            attribute consumerName = "ConsumerOne";
            attribute subscribedTopic = "TopicA";
        }

        part consumer2 : Consumer {
            attribute consumerId = "C002";
            attribute consumerName = "ConsumerTwo";
            attribute subscribedTopic = "TopicA";
        }

        part consumer3 : Consumer {
            attribute consumerId = "C003";
            attribute consumerName = "ConsumerThree";
            attribute subscribedTopic = "TopicB";
        }

        part server : Server {
            attribute serverId = "S001";
            attribute serverName = "MessageBroker";
            attribute maxTopics = 100;
        }

        connect producer1.publishPort to server.serverInputPort;
        connect producer2.publishPort to server.serverInputPort;
        connect consumer1.subscribePort to server.serverConsumerPort1;
        connect consumer2.subscribePort to server.serverConsumerPort2;
        connect consumer3.subscribePort to server.serverConsumerPort3;
    }
}