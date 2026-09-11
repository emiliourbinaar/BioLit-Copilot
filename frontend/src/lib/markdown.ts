/**
 * The smallest markdown the quoted docs need: `**bold**`, `*emphasis*` and `` `code` ``.
 *
 * Escaping happens FIRST, on the whole string, so quoted text can never inject markup. Code
 * spans are cut out before bold is applied, so nothing inside backticks is interpreted.
 */

const ESCAPES: Record<string, string> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (char) => ESCAPES[char]!);
}

export function inlineMarkdown(text: string): string {
  return text
    .split(/(`[^`]+`)/)
    .map((part) =>
      part.startsWith("`") && part.endsWith("`") && part.length > 1
        ? `<code>${escapeHtml(part.slice(1, -1))}</code>`
        : escapeHtml(part)
            .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
            // Emphasis only where the asterisks hug a word, so `2 * 3 * 4` stays arithmetic.
            .replace(/(^|[^*\w])\*(?!\s)([^*]+?)(?<!\s)\*(?![*\w])/g, "$1<em>$2</em>"),
    )
    .join("");
}
