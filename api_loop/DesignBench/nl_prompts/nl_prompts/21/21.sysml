package 'AutomatedPhotography21' {
	private import ScalarValues::*;

	part def AutomatedPhotographySystem {
		ref part user : User;
		part sceneInterface : ShootingSceneInterface;
		part focusingProcess : SceneFocusingProcess;
		part shootingProcess : ShootingProcess;
		part photoDelivery : PhotoDeliveryChannel;
		attribute userInterventionNotRequired : Boolean;
		attribute workflowSimplified : Boolean;
		attribute efficientPhotoAcquisition : Boolean;
	}

	part def ShootingSceneInterface {
		port scenePort : ScenePort;
		attribute sceneProvidedByUser : Boolean;
	}

	part def SceneFocusingProcess {
		port inputScene : ScenePort;
		port preparedImage : FocusedImagePort;
		attribute focusingCompleted : Boolean;
	}

	part def ShootingProcess {
		port inputImage : FocusedImagePort;
		port photoOutput : PhotoPort;
		attribute shootingCompleted : Boolean;
	}

	part def PhotoDeliveryChannel {
		port finalPhoto : PhotoPort;
		attribute seamlessIntegration : Boolean;
	}

	part def User {
		port receivedPhoto : PhotoPort;
	}

	port def ScenePort {
		attribute sceneData : SceneDescription;
	}

	port def FocusedImagePort {
		attribute focusedImage : FocusedImage;
	}

	port def PhotoPort {
		attribute photoData : Photograph;
	}

	attribute def SceneDescription;
	attribute def FocusedImage;
	attribute def Photograph;

	connection def SceneToFocusing {
		end fromInterface : ShootingSceneInterface::scenePort;
		end toFocusing : SceneFocusingProcess::inputScene;
	}

	connection def FocusingToShooting {
		end fromFocusing : SceneFocusingProcess::preparedImage;
		end toShooting : ShootingProcess::inputImage;
	}

	connection def ShootingToDelivery {
		end fromShooting : ShootingProcess::photoOutput;
		end toDelivery : PhotoDeliveryChannel::finalPhoto;
	}

	connection def DeliveryToUser {
		end fromDelivery : PhotoDeliveryChannel::finalPhoto;
		end toUser : User::receivedPhoto;
	}
}
