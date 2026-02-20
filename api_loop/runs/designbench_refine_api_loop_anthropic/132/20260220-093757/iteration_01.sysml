package SecurityClassificationSystem {
    private import ScalarValues::*;

    enum def ClassificationLevel {
        enum Unclassified;
        enum Confidential;
        enum Secret;
    }

    part def Component {
        attribute classificationLevel : ClassificationLevel;
        attribute label : String;
    }

    part def ConfidentialComponent {
        attribute classificationLevel : ClassificationLevel = ClassificationLevel::Confidential;
        attribute label : String = "Confidential";
        attribute displayClassification : String = "Confidential";
    }

    part def UnclassifiedComponent {
        attribute classificationLevel : ClassificationLevel = ClassificationLevel::Unclassified;
        attribute label : String = "Unclassified";
        attribute displayClassification : String = "Unclassified";
    }

    part def SecretComponent {
        attribute classificationLevel : ClassificationLevel = ClassificationLevel::Secret;
        attribute label : String = "Secret";
        attribute displayClassification : String = "Secret";
    }

    part systemRoot {
        part componentA : ConfidentialComponent {
            attribute classificationLevel : ClassificationLevel = ClassificationLevel::Confidential;
            attribute label : String = "ComponentA";
            attribute displayClassification : String = "Confidential";
        }

        part componentB : UnclassifiedComponent {
            attribute classificationLevel : ClassificationLevel = ClassificationLevel::Unclassified;
            attribute label : String = "ComponentB";
            attribute displayClassification : String = "Unclassified";
        }

        part componentC : SecretComponent {
            attribute classificationLevel : ClassificationLevel = ClassificationLevel::Secret;
            attribute label : String = "ComponentC";
            attribute displayClassification : String = "Secret";
        }

        part componentD : ConfidentialComponent {
            attribute classificationLevel : ClassificationLevel = ClassificationLevel::Confidential;
            attribute label : String = "ComponentD";
            attribute displayClassification : String = "Confidential";
        }
    }
}