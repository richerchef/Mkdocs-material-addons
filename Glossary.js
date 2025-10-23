(function () {
  async function loadGlossary() {
    const response = await fetch("glossary.json");
    return response.json();
  }

  function applyGlossary(mergedGlossary) {
    const textNodes = [];

    // Recursively find text nodes (excluding code-like elements)
    function getTextNodes(node) {
      if (
        node.nodeType === 3 &&
        node.parentNode &&
        !["CODE", "PRE", "SCRIPT", "STYLE", "SVG"].includes(node.parentNode.nodeName)
      ) {
        textNodes.push(node);
      } else {
        node.childNodes.forEach(getTextNodes);
      }
    }

    const contentRoot = document.querySelector("article");
    if (!contentRoot) return;
    getTextNodes(contentRoot);

    textNodes.forEach((textNode) => {
      let replacedText = textNode.textContent;
      for (const [term, definition] of Object.entries(mergedGlossary)) {
        const regex = new RegExp(`\\b${term}\\b`, "g");
        replacedText = replacedText.replace(
          regex,
          `<abbr title="${definition}">${term}</abbr>`
        );
      }

      if (replacedText !== textNode.textContent) {
        const span = document.createElement("span");
        span.innerHTML = replacedText;
        textNode.parentNode.replaceChild(span, textNode);
      }
    });
  }

  async function initGlossary() {
    const glossary = await loadGlossary();

    // Page-specific overrides (using abbr tags)
    const pageGlossary = {};
    document.querySelectorAll("abbr[title]").forEach((abbr) => {
      pageGlossary[abbr.textContent.trim()] = abbr.getAttribute("title");
    });

    const mergedGlossary = { ...glossary, ...pageGlossary };
    applyGlossary(mergedGlossary);
  }

  // Run when page is loaded *and* when Material reloads content
  window.addEventListener("load", initGlossary);

  // MkDocs Material instant navigation hook
  document.addEventListener("DOMContentLoaded", () => {
    if (typeof document$.subscribe === "function") {
      document$.subscribe(() => {
        initGlossary();
      });
    }
  });
})();
