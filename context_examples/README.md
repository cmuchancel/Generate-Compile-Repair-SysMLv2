# SysML Context Examples (for the agent)

Curated, one-per-grammar examples. Each subdirectory pairs a natural-language prompt (`nl.txt`) with its ground-truth SysML (`design.sysml`). Files are copied from the authoritative samples in `DesignBench/dataset/sysml/samples/`.

## How the agent should use this
- Always read *all* subdirectories before generation so the full grammar set (47/47) is in context; ingest both `nl.txt` and `design.sysml` together as paired input/output references.
- Keep pairs intact: never use `nl.txt` without the matching `design.sysml` (or vice versa).
- Use the directory name to locate the original: `NNN_Grammar_Name` → `DesignBench/dataset/sysml/samples/NN/`.
- After loading, infer the grammar patterns demonstrated in each pair and be prepared to apply any grammar type that matches the prompted NL request.
- If the user asks for structural emphasis, draw from structural grammars (`Part`, `Connection`, `Binding_Connector`, `Package`, `Port`, `Interface`, `Subsetting`, `Redefinition`).
- If the user asks for behavioral emphasis, draw from behavioral grammars (`Action`, `Action_Definition`, `Opaque_Action`, `State`, `Transition`, `State-based_Behavior`, `SequenceModeling`, `Conditional_Succession`, `Control_Structure`).
- For parametric/analytical emphasis, draw from (`Calculation`, `Constraint`, `Analysis`, `Analysis_and_Trade`, `Expression`, `Flow_Connection`).
- Default to using the full set to maximize coverage and accuracy.

## Included grammar coverage (47/47)
- Attribute — `001_Attribute`
- Generalization — `002_Generalization`
- Subsetting — `003_Subsetting`
- Redefinition — `004_Redefinition`
- Enumeration — `005_Enumeration`
- Part — `007_Part`
- Item — `008_Item`
- Connection — `009_Connection`
- Port — `010_Port`
- Function-based Behavior — `015_Function-based_Behavior`
- Interface — `021_Interface`
- State-based Behavior — `024_State-based_Behavior`
- Individual and Snapshot — `026_Individual_and_Snapshot`
- Variant Configuration — `027_Variant_Configuration`
- Requirement — `030_Requirement`
- Verification — `031_Verification`
- Analysis and Trade — `033_Analysis_and_Trade`
- View and Viewpoint — `037_View_and_Viewpoint`
- Dependency — `039_Dependency`
- Model Constrainment — `042_Model_Constrainment`
- Binding Connector — `043_Binding_Connector`
- Language Extension — `046_Language_Extension`
- Expression — `049_Expression`
- SequenceModeling — `064_SequenceModeling`
- Flow Connection — `065_Flow_Connection`
- Action Definition — `070_Action_Definition`
- Action — `074_Action`
- Conditional Succession — `075_Conditional_Succession`
- Control Structure — `077_Control_Structure`
- Assignment Action — `083_Assignment_Action`
- Message — `084_Message`
- Opaque Action — `086_Opaque_Action`
- State Definition — `087_State_Definition`
- State — `089_State`
- Transition — `092_Transition`
- Occurrence — `096_Occurrence`
- Individual — `103_Individual`
- Calculation — `110_Calculation`
- Constraint — `113_Constraint`
- Analysis — `124_Analysis`
- Use Case — `129_Use_Case`
- Variability — `131_Variability`
- Functional Allocation — `135_Functional_Allocation`
- Metadata — `137_Metadata`
- Filtering — `139_Filtering`
- View — `144_View`
- Package — `146_Package`

## Tips
- For quick experiments, select 5–10 grammars aligned to the task; for maximum fidelity, load all 47 pairs.
- Traceback: the `NN` prefix maps directly to `DesignBench/dataset/sysml/samples/NN/`.
