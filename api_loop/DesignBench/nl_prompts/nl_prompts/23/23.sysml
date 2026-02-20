package 'PhotographyWorkflow23' {
	private import ScalarValues::*;

	part def PhotographyWorkflowSystem {
		ref part user : User;
		part sceneSelection : SceneSelectionInterface;
		part focusingProcess : FocusingProcess;
		part focusAssessment : FocusAssessment;
		part shootingProcess : ShootingProcess;
		part photoOutput : PhotoOutputInterface;
		attribute oneClickExperienceProvided : Boolean;
		attribute automaticSequenceEnsured : Boolean;
		attribute clarityGuaranteed : Boolean;
		attribute qualityMaintained : Boolean;
	}

	part def SceneSelectionInterface {
		port scenePort : ScenePort;
	}

	part def FocusingProcess {
		port inputScene : ScenePort;
		port focusedImage : FocusedImagePort;
		attribute optimalSharpnessTargeted : Boolean;
	}

	part def FocusAssessment {
		port imageToAssess : FocusedImagePort;
		attribute focusClearDetected : Boolean;
	}

	part def ShootingProcess {
		port shootingInput : FocusedImagePort;
		port producedPhoto : PhotoPort;
		attribute photoGenerated : Boolean;
	}

	part def PhotoOutputInterface {
		port deliveredPhoto : PhotoPort;
	}

	part def User {
		port photoReception : PhotoPort;
	}

	port def ScenePort {
		attribute sceneData : SceneDescription;
	}

	port def FocusedImagePort {
		attribute imageData : FocusedImage;
	}

	port def PhotoPort {
		attribute photoData : Photograph;
	}

	attribute def SceneDescription;
	attribute def FocusedImage;
	attribute def Photograph;

	connection def SceneToFocusing {
		end fromSelection : SceneSelectionInterface::scenePort;
		end toFocusing : FocusingProcess::inputScene;
	}

	connection def FocusingToAssessment {
		end fromFocusing : FocusingProcess::focusedImage;
		end toAssessment : FocusAssessment::imageToAssess;
	}

	connection def AssessmentToShooting {
		end fromAssessment : FocusAssessment::imageToAssess;
		end toShooting : ShootingProcess::shootingInput;
	}

	connection def ShootingToOutput {
		end fromShooting : ShootingProcess::producedPhoto;
		end toOutput : PhotoOutputInterface::deliveredPhoto;
	}

	connection def OutputToUser {
		end fromOutput : PhotoOutputInterface::deliveredPhoto;
		end toUser : User::photoReception;
	}
}
