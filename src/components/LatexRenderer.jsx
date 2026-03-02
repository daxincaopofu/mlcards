import katex from "katex";
import "katex/dist/katex.min.css";

export default function LatexRenderer({ text, style = {}, serif = false }) {
  const segments = [];
  const re = /(\$\$[\s\S]+?\$\$|\$[^$\n]+?\$)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) segments.push({ type: "text", content: text.slice(last, m.index) });
    const isDisplay = m[0].startsWith("$$");
    const inner = isDisplay ? m[0].slice(2, -2) : m[0].slice(1, -1);
    segments.push({ type: "latex", display: isDisplay, content: inner });
    last = m.index + m[0].length;
  }
  if (last < text.length) segments.push({ type: "text", content: text.slice(last) });

  return (
    <span style={style}>
      {segments.map((seg, i) => {
        if (seg.type === "text") {
          return (
            <span key={i} style={{ whiteSpace: "pre-wrap", fontFamily: serif ? "'Cormorant Garamond', serif" : "inherit" }}>
              {seg.content}
            </span>
          );
        }
        try {
          const html = katex.renderToString(seg.content, {
            displayMode: seg.display,
            throwOnError: false,
            output: "html",
          });
          return (
            <span key={i}
              style={{ display: seg.display ? "block" : "inline", textAlign: seg.display ? "center" : "inherit", margin: seg.display ? "10px 0" : "0 2px" }}
              dangerouslySetInnerHTML={{ __html: html }}
            />
          );
        } catch {
          return <span key={i}>{seg.content}</span>;
        }
      })}
    </span>
  );
}
