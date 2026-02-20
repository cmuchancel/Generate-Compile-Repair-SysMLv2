package 'PhotographyWorkflow19' {
	private import ScalarValues::*;

	part def PhotographyWorkflowSystem {
		ref part user : User;
		part sceneSetup : SceneConfiguration;
		part focusingStep : FocusingStep;
		part shootingStep : ShootingStep;
		part photoOutput : PhotoDeliveryChannel;
		attribute automaticFocusing : Boolean;
		attribute automaticShooting : Boolean;
		attribute focusAchievedBeforeShooting : Boolean;
		attribute highQualityPhotoOutput : Boolean;
		attribute workflowConcise : Boolean;
		attribute workflowCoherent : Boolean;
		attribute efficientAcquisition : Boolean;
	}

	part def SceneConfiguration {
		port selectedScenePort : ScenePort;
	}

	part def FocusingStep {
		port sceneInputPort : ScenePort;
		port focusedImagePort : FocusedImagePort;
		attribute intermediateImageGenerated : Boolean;
	}

	part def ShootingStep {
		port focusedImageInput : FocusedImagePort;
		port photoOutputPort : PhotoPort;
	}

	part def PhotoDeliveryChannel {
		port deliveredPhotoPort : PhotoPort;
	}

	part def User {
		port receivedPhotoPort : PhotoPort;
	}

	port def ScenePort {
		attribute sceneData : SceneDescription;
	}

	port def FocusedImagePort {
		attribute focusedImage : FocusedImage;
	}

	port def PhotoPort {
		attribute photoData : CapturedPhoto;
	}

	attribute def SceneDescription;
	attribute def FocusedImage;
	attribute def CapturedPhoto;

	connection def SceneSelectionFlow {
		end fromSceneSetup : SceneConfiguration::selectedScenePort;
		end toFocusing : FocusingStep::sceneInputPort;
	}

	connection def FocusingFlow {
		end fromFocusing : FocusingStep::focusedImagePort;
		end toShooting : ShootingStep::focusedImageInput;
	}

	connection def ShootingFlow {
		end fromShooting : ShootingStep::photoOutputPort;
		end toDelivery : PhotoDeliveryChannel::deliveredPhotoPort;
	}

	connection def DeliveryFlow {
		end fromDelivery : PhotoDeliveryChannel::deliveredPhotoPort;
		end toUser : User::receivedPhotoPort;
	}
}
