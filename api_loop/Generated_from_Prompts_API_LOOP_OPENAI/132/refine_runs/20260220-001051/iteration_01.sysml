package SecurityModel {
  private import ScalarValues::*;
  
  enum def SecurityClassification {
    enum Unclassified;
    enum Confidential;
    enum Secret;
  }
  
  part def Component {
    attribute classification : SecurityClassification;
  }
  
  part def UnclassifiedComponent : Component {
    attribute classification : SecurityClassification = Unclassified;
  }
  
  part def ConfidentialComponent : Component {
    attribute classification : SecurityClassification = Confidential;
  }
  
  part def SecretComponent : Component {
    attribute classification : SecurityClassification = Secret;
  }
  
  part def System {
    part compA : UnclassifiedComponent;
    part compB : ConfidentialComponent;
    part compC : SecretComponent;
  }
}