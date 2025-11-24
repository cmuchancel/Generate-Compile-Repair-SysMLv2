package 'PubSubMessageFlow' {
	private import ScalarValues::*;
	item def Topic {
		attribute name : String;
	}
	item def TopicMessage {
		attribute topicName : String;
		attribute content : String;
	}
	part def Producer {
		attribute name : String;
	}
	part def Consumer {
		attribute name : String;
	}
	part def TopicSubscription {
		ref part consumer : Consumer;
		ref part topic : Topic;
	}
	part def Server {
		part subscriptions : TopicSubscription[*];
	}
	action def SendSubscriptionRequest {
		in consumer : Consumer;
		in topic : Topic;
		out subscription : TopicSubscription;
	}
	action def PublishTopicMessage {
		in producer : Producer;
		in topic : Topic;
		out producedPayload : TopicMessage;
	}
	action def PushMessageToSubscriber {
		in server : Server;
		in topic : Topic;
		in subscription : TopicSubscription;
		in publication : TopicMessage;
		out deliveredPayload : TopicMessage;
	}
	part pubSubProcess {
		part producer : Producer {
			attribute name = "DevicePublisher";
		}
		part server : Server;
		part analyticsConsumer : Consumer {
			attribute name = "AnalyticsClient";
		}
		part auditConsumer : Consumer {
			attribute name = "AuditConsole";
		}
		part systemTopic : Topic {
			attribute name = "SystemTopic";
		}
		part publishedPayload : TopicMessage {
			attribute topicName = systemTopic.name;
			attribute content = "New telemetry sample";
		}
		part analyticsSubscription : TopicSubscription {
			ref part consumer = pubSubProcess::analyticsConsumer;
			ref part topic = systemTopic;
		}
		part auditSubscription : TopicSubscription {
			ref part consumer = pubSubProcess::auditConsumer;
			ref part topic = systemTopic;
		}
		action analyticsSubscribe : SendSubscriptionRequest {
			in consumer = pubSubProcess::analyticsConsumer;
			in topic = systemTopic;
			out subscription = analyticsSubscription;
		}
		action auditSubscribe : SendSubscriptionRequest {
			in consumer = pubSubProcess::auditConsumer;
			in topic = systemTopic;
			out subscription = auditSubscription;
		}
		action publishAction : PublishTopicMessage {
			in producer = pubSubProcess::producer;
			in topic = systemTopic;
			out producedPayload = publishedPayload;
		}
		action pushAnalytics : PushMessageToSubscriber {
			in server = pubSubProcess::server;
			in topic = systemTopic;
			in subscription = analyticsSubscription;
			in publication = publishedPayload;
			out deliveredPayload = publishedPayload;
		}
		action pushAudit : PushMessageToSubscriber {
			in server = pubSubProcess::server;
			in topic = systemTopic;
			in subscription = auditSubscription;
			in publication = publishedPayload;
			out deliveredPayload = publishedPayload;
		}
	}
	requirement def 'ReliableDistributionRequirement' {
		text = "Ensure producers publish topic messages to the server, consumers subscribe via the server, and every new message is pushed to all subscribers of the topic.";
	}
}
