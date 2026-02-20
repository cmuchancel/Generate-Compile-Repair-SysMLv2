package PubSubSystem {
    private import ScalarValues::*;

    enum def TopicStatus {
        enum ACTIVE;
        enum INACTIVE;
    }

    enum def DeliveryStatus {
        enum PENDING;
        enum DELIVERED;
        enum FAILED;
    }

    item def Message {
        attribute messageId : String;
        attribute topicName : String;
        attribute content : String;
        attribute timestamp : String;
    }

    item def TopicSubscription {
        attribute topicName : String;
        attribute subscriberId : String;
        attribute status : TopicStatus;
    }

    item def PublishRequest {
        attribute topicName : String;
        attribute publisherId : String;
        attribute payload : String;
    }

    item def DeliveryRecord {
        attribute messageId : String;
        attribute subscriberId : String;
        attribute deliveryStatus : DeliveryStatus;
    }

    port def ProducerOutPort {
        attribute topicName : String;
        attribute publisherId : String;
    }

    port def ServerInFromProducerPort {
        attribute receivedTopic : String;
        attribute receivedPublisherId : String;
    }

    port def ServerOutToConsumerPort {
        attribute deliveredTopic : String;
        attribute targetSubscriberId : String;
    }

    port def ConsumerInPort {
        attribute subscribedTopic : String;
        attribute consumerId : String;
    }

    port def ConsumerSubscribePort {
        attribute topicOfInterest : String;
        attribute consumerId : String;
    }

    port def ServerSubscriptionPort {
        attribute registeredTopic : String;
        attribute registeredSubscriberId : String;
    }

    part def Producer {
        attribute producerId : String;
        attribute topicToPublish : String;
        attribute messageContent : String;

        port publishPort : ProducerOutPort;
    }

    part def Broker {
        attribute brokerId : String;
        attribute brokerName : String;
        attribute maxTopics : Integer;

        port receiveFromProducerPort : ServerInFromProducerPort;
        port sendToConsumerPort : ServerOutToConsumerPort;
        port subscriptionManagementPort : ServerSubscriptionPort;
    }

    part def Consumer {
        attribute consumerId : String;
        attribute subscribedTopic : String;
        attribute receivedMessageContent : String;

        port subscribePort : ConsumerSubscribePort;
        port receivePort : ConsumerInPort;
    }

    part def TopicRegistry {
        attribute registryId : String;
        attribute totalTopics : Integer;

        attribute topic1 : String;
        attribute topic2 : String;
        attribute topic3 : String;
        attribute topic4 : String;
        attribute topic5 : String;
    }

    part def SubscriptionManager {
        attribute managerId : String;
        attribute totalSubscriptions : Integer;
    }

    part def MessageDispatcher {
        attribute dispatcherId : String;
        attribute pendingMessages : Integer;
        attribute dispatchedMessages : Integer;
    }

    part def PubSubPlatform {
        attribute platformId : String;
        attribute platformName : String;
        attribute isOperational : Boolean;

        part producer1 : Producer {
            attribute producerId = "P001";
            attribute topicToPublish = "TOPIC_NEWS";
            attribute messageContent = "Breaking News Update";
        }

        part producer2 : Producer {
            attribute producerId = "P002";
            attribute topicToPublish = "TOPIC_SPORTS";
            attribute messageContent = "Sports Score Update";
        }

        part broker : Broker {
            attribute brokerId = "B001";
            attribute brokerName = "MainBroker";
            attribute maxTopics = 100;
        }

        part consumer1 : Consumer {
            attribute consumerId = "C001";
            attribute subscribedTopic = "TOPIC_NEWS";
            attribute receivedMessageContent = "";
        }

        part consumer2 : Consumer {
            attribute consumerId = "C002";
            attribute subscribedTopic = "TOPIC_SPORTS";
            attribute receivedMessageContent = "";
        }

        part consumer3 : Consumer {
            attribute consumerId = "C003";
            attribute subscribedTopic = "TOPIC_NEWS";
            attribute receivedMessageContent = "";
        }

        part topicRegistry : TopicRegistry {
            attribute registryId = "TR001";
            attribute totalTopics = 5;
            attribute topic1 = "TOPIC_NEWS";
            attribute topic2 = "TOPIC_SPORTS";
            attribute topic3 = "TOPIC_FINANCE";
            attribute topic4 = "TOPIC_TECH";
            attribute topic5 = "TOPIC_WEATHER";
        }

        part subscriptionManager : SubscriptionManager {
            attribute managerId = "SM001";
            attribute totalSubscriptions = 3;
        }

        part messageDispatcher : MessageDispatcher {
            attribute dispatcherId = "MD001";
            attribute pendingMessages = 0;
            attribute dispatchedMessages = 0;
        }

        connect producer1.publishPort to broker.receiveFromProducerPort;
        connect producer2.publishPort to broker.receiveFromProducerPort;
        connect consumer1.subscribePort to broker.subscriptionManagementPort;
        connect consumer2.subscribePort to broker.subscriptionManagementPort;
        connect consumer3.subscribePort to broker.subscriptionManagementPort;
        connect broker.sendToConsumerPort to consumer1.receivePort;
        connect broker.sendToConsumerPort to consumer2.receivePort;
        connect broker.sendToConsumerPort to consumer3.receivePort;
    }
}