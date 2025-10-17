document.addEventListener("DOMContentLoaded", async () => {
  try {
    const response = await fetch("/glossary.json");
    const globalGlossary = await response.json();

    const contentEl = document.querySelector(".md-content__inner");
    if (!contentEl) return;

    // 1. Extract per-page glossary overrides
    const html = contentEl.innerHTML;
    const pageGlossary = {};
    const regex = /\*\[([^\]]+)\]:\s*(.+)/g;
    let match;
    while ((match = regex.exec(html)) !== null) {
      pageGlossary[match[1].trim()] = match[2].trim();
    }

    // Remove reference lines
    contentEl.innerHTML = html.replace(regex, "");

    // Merge global + page-specific (page overrides global)
    const glossary = { ...globalGlossary, ...pageGlossary };

    // 2. Walk the DOM text nodes and replace only text outside <abbr> or <code>
    const walker = document.createTreeWalker(contentEl, NodeFilter.SHOW_TEXT, null);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);

    nodes.forEach(node => {
      if (node.parentNode.tagName === "ABBR" || node.parentNode.tagName === "CODE") return;

      let text = node.nodeValue;
      Object.entries(glossary).forEach(([term, def]) => {
        const regex = new RegExp(`\\b(${term})\\b`, "g");
        text = text.replace(regex, `<abbr title="${def}" class="glossary-term">$1</abbr>`);
      });

      if (text !== node.nodeValue) {
        const span = document.createElement("span");
        span.innerHTML = text;
        node.replaceWith(span);
      }
    });

  } catch (err) {
    console.error("Glossary loading error:", err);
  }
});
