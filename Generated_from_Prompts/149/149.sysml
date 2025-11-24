package 'PublishSubscribeMessaging' {
	private import ScalarValues::*;
	item def Topic {
		attribute name : String;
	}
item def PublishedMessage {
	attribute topicName : String;
	attribute payload : String;
}
	port def PublishPort {
		out item publication : PublishedMessage;
	}
	port def ServerIntakePort {
		in item publication : PublishedMessage;
	}
	port def DeliveryPort {
		out item payload : PublishedMessage;
	}
	port def InboxPort {
		in item payload : PublishedMessage;
	}
	port def SubscriptionDeclarationPort {
		out item topic : Topic;
	}
	port def SubscriptionReceptionPort {
		in item topic : Topic;
	}
	part def Producer {
		attribute name : String;
		port publish : PublishPort;
	}
	part def Subscriber {
		attribute name : String;
		port topicDeclaration : SubscriptionDeclarationPort;
		port inbox : InboxPort;
	}
	part def Subscription {
		ref part subscriber : Subscriber;
		ref part topic : Topic;
	}
	part def Server {
		port intake : ServerIntakePort;
		port subscriptionReception : SubscriptionReceptionPort;
		port delivery : DeliveryPort;
		part subscriptions : Subscription[*];
	}
	connection def PublishChannel {
		end publisher : PublishPort[1];
		end server : ServerIntakePort[1];
	}
	connection def SubscriptionChannel {
		end subscriber : SubscriptionDeclarationPort[1];
		end server : SubscriptionReceptionPort[1];
	}
	connection def DeliveryChannel {
		end server : DeliveryPort[1];
		end subscriber : InboxPort[1];
	}
	action def DeclareSubscription {
		in subscriber : Subscriber;
		in topic : Topic;
		out subscription : Subscription;
	}
	action def PublishToServer {
		in producer : Producer;
		in topic : Topic;
		in publication : PublishedMessage;
		out publishedPayload : PublishedMessage;
	}
	action def DistributeToSubscriber {
		in server : Server;
		in subscription : Subscription;
		in publication : PublishedMessage;
		out deliveredPayload : PublishedMessage;
	}
	part pubSubModel {
		part engineeringTopic : Topic {
			attribute name = "System Alerts";
		}
		part producer : Producer {
			attribute name = "TelemetryPublisher";
		}
		part subscriber : Subscriber {
			attribute name = "MonitoringConsole";
		}
		part server : Server;
		part declaredSubscription : Subscription {
			ref part subscriber = pubSubModel::subscriber;
			ref part topic = pubSubModel::engineeringTopic;
		}
		part initialMessage : PublishedMessage {
			attribute topicName = engineeringTopic.name;
			attribute payload = "Initial diagnostics";
		}
		connection publishLink : PublishChannel {
			end publisher = producer.publish;
			end serverIntake = server.intake;
		}
		connection subscriptionLink : SubscriptionChannel {
			end subscriberDeclaration = subscriber.topicDeclaration;
			end registry = server.subscriptionReception;
		}
		connection deliveryLink : DeliveryChannel {
			end dispatcher = server.delivery;
			end inboxPort = subscriber.inbox;
		}
		action subscriptionAction : DeclareSubscription {
			in subscriber = pubSubModel::subscriber;
			in topic = engineeringTopic;
			out subscription = declaredSubscription;
		}
		action publishAction : PublishToServer {
			in producer = pubSubModel::producer;
			in topic = engineeringTopic;
			in publication = initialMessage;
			out publishedPayload = initialMessage;
		}
		action distributeAction : DistributeToSubscriber {
			in server = pubSubModel::server;
			in subscription = declaredSubscription;
			in publication = initialMessage;
			out deliveredPayload = initialMessage;
		}
	}
	requirement def 'Topic Declaration Requirement' {
		text = "Subscribers shall declare the topics they follow without needing to know publishers or other consumers.";
	}
	requirement def 'PubSubProcessingRequirement' {
		text = "The server shall manage subscription, publishing, and distribution flows so that published messages are promptly routed to interested subscribers.";
	}
}
