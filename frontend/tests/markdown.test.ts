import { describe, expect, it } from "vitest";
import { inlineMarkdown } from "../src/lib/markdown";

describe("inlineMarkdown", () => {
  it("renders bold and code, escapes everything else, and never interprets inside code", () => {
    // Quoted docs carry `**bold**` and `code`, and fixture strings are data. Escaping comes
    // first so no quoted text can inject markup into the page.
    expect(inlineMarkdown("a **b** `c<d>` & e")).toBe(
      "a <strong>b</strong> <code>c&lt;d&gt;</code> &amp; e",
    );
    expect(inlineMarkdown("`**not bold**`")).toBe("<code>**not bold**</code>");
    expect(inlineMarkdown('<script>alert("x")</script>')).toBe(
      "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;",
    );
    // The retrospective writes emphasis as *single asterisks*; left literal, they printed as
    // stray punctuation inside quoted text on the home page.
    expect(inlineMarkdown("scores *below* it")).toBe("scores <em>below</em> it");
    expect(inlineMarkdown("2 * 3 * 4 and `*x*`")).toBe("2 * 3 * 4 and <code>*x*</code>");
  });
});
