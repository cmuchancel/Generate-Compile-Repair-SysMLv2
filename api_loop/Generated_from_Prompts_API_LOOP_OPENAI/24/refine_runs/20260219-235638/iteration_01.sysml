package PhotoWorkflow {
  private import ScalarValues::*;

  enum def FocusQuality { Good; Bad; }

  item def Scene;
  item def Image;
  item def Photograph;
  item def FocusEvaluation { attribute quality: FocusQuality; }

  port def SceneInPort { item scene: Scene; }
  port def SceneOutPort { item scene: Scene; }
  port def ImagePort { item image: Image; }
  port def EvalPort { item evaluation: FocusEvaluation; }
  port def PhotoPort { item photo: Photograph; }

  part def SceneSelector {
    port sceneOut: SceneOutPort;
  }

  part def FocusAndImage {
    port sceneIn: SceneInPort;
    port imageOut: ImagePort;
  }

  part def FocusEvaluator {
    port imageIn: ImagePort;
    port evalOut: EvalPort;
  }

  part def Shooter {
    port imageIn: ImagePort;
    port evalIn: EvalPort;
    port photoOut: PhotoPort;
  }

  part def PhotoOutput {
    port photoIn: PhotoPort;
  }

  part def CameraSystem {
    part selector: SceneSelector;
    part focus: FocusAndImage;
    part evaluator: FocusEvaluator;
    part shooter: Shooter;
    part output: PhotoOutput;

    connect selector.sceneOut to focus.sceneIn;
    connect focus.imageOut to evaluator.imageIn;
    connect evaluator.evalOut to shooter.evalIn;
    connect focus.imageOut to shooter.imageIn;
    connect shooter.photoOut to output.photoIn;
  }
}