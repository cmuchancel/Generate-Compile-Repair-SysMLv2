package 'PhotographyWorkflow24' {
	private import ScalarValues::*;

	part def PhotographyWorkflowSystem {
		ref part user : User;
		part sceneInput : SceneInputInterface;
		part focusingStage : FocusingStage;
		part focusEvaluation : FocusEvaluation;
		part shootingStage : ShootingStage;
		part photoOutput : PhotoOutputInterface;
		attribute automaticLinkingProvided : Boolean;
		attribute focusQualityGateEnforced : Boolean;
		attribute clarityImproved : Boolean;
		attribute qualityImproved : Boolean;
	}

	part def SceneInputInterface {
		port providedScene : ScenePort;
	}

	part def FocusingStage {
		port inputScene : ScenePort;
		port focusedImage : FocusedImagePort;
	}

	part def FocusEvaluation {
		port evaluatedImage : FocusedImagePort;
		attribute focusQualityAcceptable : Boolean;
	}

	part def ShootingStage {
		port shootingInput : FocusedImagePort;
		port generatedPhoto : PhotoPort;
		attribute shootingExecuted : Boolean;
	}

	part def PhotoOutputInterface {
		port deliveredPhoto : PhotoPort;
	}

	part def User {
		port receivedPhoto : PhotoPort;
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

	connection def SceneToFocus {
		end fromInput : SceneInputInterface::providedScene;
		end toFocusing : FocusingStage::inputScene;
	}

	connection def FocusToEvaluation {
		end fromFocusing : FocusingStage::focusedImage;
		end toEvaluation : FocusEvaluation::evaluatedImage;
	}

	connection def EvaluationToShooting {
		end fromEvaluation : FocusEvaluation::evaluatedImage;
		end toShooting : ShootingStage::shootingInput;
	}

	connection def ShootingToOutput {
		end fromShooting : ShootingStage::generatedPhoto;
		end toOutput : PhotoOutputInterface::deliveredPhoto;
	}

	connection def OutputToUser {
		end fromOutput : PhotoOutputInterface::deliveredPhoto;
		end toUser : User::receivedPhoto;
	}
}
