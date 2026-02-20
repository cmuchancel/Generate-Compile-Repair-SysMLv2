package 'PhotographyWorkflow25' {
	private import ScalarValues::*;

	part def PhotographyWorkflowSystem {
		ref part user : User;
		part triggerInterface : PhotoTriggerInterface;
		part sceneAcquisition : SceneAcquisition;
		part focusFunction : FocusFunction;
		part shootingOperation : ShootingOperation;
		part photoDisplay : PhotoDisplay;
		attribute singleTriggerSufficient : Boolean;
		attribute seamlessIntegrationProvided : Boolean;
		attribute convenienceImproved : Boolean;
		attribute smoothnessImproved : Boolean;
	}

	part def PhotoTriggerInterface {
		port takePhotoCommand : TriggerPort;
		attribute triggerIssuedByUser : Boolean;
	}

	part def SceneAcquisition {
		port acquisitionTrigger : TriggerPort;
		port capturedScene : ScenePort;
		attribute currentSceneObtained : Boolean;
	}

	part def FocusFunction {
		port sceneInput : ScenePort;
		port generatedImage : ImagePort;
		attribute focusingCompleted : Boolean;
	}

	part def ShootingOperation {
		port imageInput : ImagePort;
		port producedPhoto : PhotoPort;
		attribute shootingCompleted : Boolean;
	}

	part def PhotoDisplay {
		port displayedPhoto : PhotoPort;
		attribute immediateDisplayProvided : Boolean;
	}

	part def User {
		port shootingTrigger : TriggerPort;
		port viewedPhoto : PhotoPort;
	}

	port def TriggerPort {
		attribute triggerSignal : Boolean;
	}

	port def ScenePort {
		attribute sceneData : SceneDescription;
	}

	port def ImagePort {
		attribute imageData : FocusedImage;
	}

	port def PhotoPort {
		attribute photoData : Photograph;
	}

	attribute def SceneDescription;
	attribute def FocusedImage;
	attribute def Photograph;

	connection def UserTrigger {
		end fromUser : User::shootingTrigger;
		end toInterface : PhotoTriggerInterface::takePhotoCommand;
	}

	connection def TriggerToAcquisition {
		end fromTrigger : PhotoTriggerInterface::takePhotoCommand;
		end toAcquisition : SceneAcquisition::acquisitionTrigger;
	}

	connection def SceneToFocus {
		end fromAcquisition : SceneAcquisition::capturedScene;
		end toFocus : FocusFunction::sceneInput;
	}

	connection def FocusToShooting {
		end fromFocus : FocusFunction::generatedImage;
		end toShooting : ShootingOperation::imageInput;
	}

	connection def ShootingToDisplay {
		end fromShooting : ShootingOperation::producedPhoto;
		end toDisplay : PhotoDisplay::displayedPhoto;
	}

	connection def DisplayToUser {
		end fromDisplay : PhotoDisplay::displayedPhoto;
		end toUser : User::viewedPhoto;
	}
}
