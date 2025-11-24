package 'AutomobileDocumentPlatform' {
	private import ScalarValues::*;
	public import ISQ::TorqueValue;
	part def Automobile;
	public alias Car for Automobile;
	attribute def DocumentRecord {
		attribute subjectLabel : String;
		attribute explanatoryText : String;
	}
	action def EnterAutomobileInfo {
		in automobile : Automobile;
		in detail : String;
	}
	action def QueryAutomobileRecords {
		in automobile : Automobile;
		out summary : String;
	}
	action def ArchiveAutomobileDocument {
		in automobile : Automobile;
		in document : DocumentRecord;
	}
	part documentPlatform {
		part managedAutomobile : Automobile;
		attribute torqueReference : TorqueValue;
		action enterInfo : EnterAutomobileInfo {
			in automobile = managedAutomobile;
		}
		action queryRecords : QueryAutomobileRecords {
			in automobile = managedAutomobile;
		}
		action archiveRecord : ArchiveAutomobileDocument {
			in automobile = managedAutomobile;
		}
		attribute automobileDocumentation : DocumentRecord {
			attribute redefines subjectLabel = "Automobile";
		}
		attribute carAliasDocumentation : DocumentRecord {
			attribute redefines subjectLabel = "Car";
		}
	}
}
