import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.antlr.v4.runtime.BaseErrorListener;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.Recognizer;
import org.sireum.hamr.sysml.parser.SysMLv2Lexer;
import org.sireum.hamr.sysml.parser.SysMLv2Parser;

public final class ParseSysML {
  private static final class CollectingErrorListener extends BaseErrorListener {
    private final List<String> errors = new ArrayList<>();

    @Override
    public void syntaxError(
        Recognizer<?, ?> recognizer,
        Object offendingSymbol,
        int line,
        int charPositionInLine,
        String msg,
        RecognitionException e) {
      errors.add("line " + line + ":" + charPositionInLine + " " + msg);
    }

    public List<String> getErrors() {
      return errors;
    }
  }

  public static void main(String[] args) throws Exception {
    if (args.length != 1) {
      System.err.println("Usage: ParseSysML <file.sysml>");
      System.exit(2);
    }

    String file = Path.of(args[0]).toString();
    CharStream input = CharStreams.fromFileName(file);

    SysMLv2Lexer lexer = new SysMLv2Lexer(input);
    CollectingErrorListener lexErr = new CollectingErrorListener();
    lexer.removeErrorListeners();
    lexer.addErrorListener(lexErr);

    CommonTokenStream tokens = new CommonTokenStream(lexer);
    SysMLv2Parser parser = new SysMLv2Parser(tokens);
    CollectingErrorListener parseErr = new CollectingErrorListener();
    parser.removeErrorListeners();
    parser.addErrorListener(parseErr);

    parser.entryRuleRootNamespace();

    List<String> all = new ArrayList<>();
    all.addAll(lexErr.getErrors());
    all.addAll(parseErr.getErrors());

    if (!all.isEmpty() || parser.getNumberOfSyntaxErrors() > 0) {
      System.out.println("ANTLR_PARSE_FAIL " + file);
      for (String err : all) {
        System.out.println(err);
      }
      System.exit(1);
    }

    System.out.println("ANTLR_PARSE_PASS " + file);
  }
}
