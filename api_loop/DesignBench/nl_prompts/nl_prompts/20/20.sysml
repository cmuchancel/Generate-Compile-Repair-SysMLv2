package 'PhotographyWorkflow20' {
	private import ScalarValues::*;

	part def PhotographyWorkflowSystem {
		ref part user : User;
		part sceneFraming : SceneFramingInterface;
		part focusingProcess : SceneFocusingProcess;
		part shootingProcess : ShootingProcess;
		part photoDelivery : PhotoDeliveryInterface;
		attribute automaticFocusingProvided : Boolean;
		attribute automaticShootingProvided : Boolean;
		attribute intermediateProcessingAutomated : Boolean;
		attribute highQualityPhotographsEnsured : Boolean;
		attribute convenientSceneConversion : Boolean;
	}

	part def SceneFramingInterface {
		port framedScenePort : ScenePort;
		attribute sceneSelectionSupported : Boolean;
	}

	part def SceneFocusingProcess {
		port inputScenePort : ScenePort;
		port generatedImagePort : FocusedImagePort;
		attribute focusingStepCompleted : Boolean;
	}

	part def ShootingProcess {
		port imageInputPort : FocusedImagePort;
		port photoOutputPort : PhotoPort;
		attribute shootingStepCompleted : Boolean;
	}

	part def PhotoDeliveryInterface {
		port deliveredPhotoPort : PhotoPort;
	}

	part def User {
		port photoReceiptPort : PhotoPort;
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

	connection def SceneFramingToFocusing {
		end fromFraming : SceneFramingInterface::framedScenePort;
		end toFocusing : SceneFocusingProcess::inputScenePort;
	}

	connection def FocusingToShooting {
		end focusedImage : SceneFocusingProcess::generatedImagePort;
		end shootingInput : ShootingProcess::imageInputPort;
	}

	connection def ShootingToDelivery {
		end fromShooting : ShootingProcess::photoOutputPort;
		end toDelivery : PhotoDeliveryInterface::deliveredPhotoPort;
	}

	connection def DeliveryToUser {
		end fromDelivery : PhotoDeliveryInterface::deliveredPhotoPort;
		end toUser : User::photoReceiptPort;
	}
}
