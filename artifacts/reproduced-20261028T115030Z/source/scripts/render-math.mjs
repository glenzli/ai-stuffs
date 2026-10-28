import katex from "katex";
import { parseFragment } from "parse5";

export function renderMath(content, file = "content") {
  const fragment = parseFragment(content, { sourceCodeLocationInfo: true });
  const replacements = [];
  function visit(node) {
    const marker = node.attrs?.find(attr => attr.name === "data-math");
    if (marker) {
      if (!["div", "span"].includes(node.tagName) || !node.sourceCodeLocation?.endTag) {
        throw new Error(`${file}: data-math requires a closed div or span`);
      }
      if (node.childNodes.some(child => child.nodeName !== "#text")) {
        throw new Error(`${file}: math source must be plain TeX text; escape HTML < and &`);
      }
      const tex = node.childNodes.map(child => child.value).join("").trim();
      if (!tex) throw new Error(`${file}: empty math expression`);
      if (!["", "display", "inline"].includes(marker.value)) throw new Error(`${file}: invalid math mode ${marker.value}`);
      let rendered;
      try {
        rendered = katex.renderToString(tex, {
          displayMode: marker.value !== "inline",
          output: "htmlAndMathml",
          throwOnError: true,
          strict: "error",
          trust: false,
          maxExpand: 1000,
        });
      } catch (error) {
        throw new Error(`${file}:${node.sourceCodeLocation.startLine}: ${error.message}`);
      }
      replacements.push({ start: node.sourceCodeLocation.startTag.endOffset, end: node.sourceCodeLocation.endTag.startOffset, rendered });
      return;
    }
    node.childNodes?.forEach(visit);
    if (node.content) visit(node.content);
  }
  visit(fragment);
  let html = content;
  for (const replacement of replacements.sort((a, b) => b.start - a.start)) {
    html = html.slice(0, replacement.start) + replacement.rendered + html.slice(replacement.end);
  }
  return { html, count: replacements.length };
}
