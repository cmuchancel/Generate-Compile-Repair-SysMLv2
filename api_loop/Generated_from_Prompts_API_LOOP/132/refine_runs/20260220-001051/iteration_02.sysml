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

  part def UnclassifiedComponent {
    attribute classification : SecurityClassification = SecurityClassification::Unclassified;
  }

  part def ConfidentialComponent {
    attribute classification : SecurityClassification = SecurityClassification::Confidential;
  }

  part def SecretComponent {
    attribute classification : SecurityClassification = SecurityClassification::Secret;
  }

  part def System {
    part compA : UnclassifiedComponent;
    part compB : ConfidentialComponent;
    part compC : SecretComponent;
  }
}