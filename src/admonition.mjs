// tohoku-admonition.mjs
/**
 * A custom admonition styled with Tohoku University's purple color
 * and always shown as a dropdown with a question mark icon.
 */
const myAdmonition = {
  name: "checkitout",
  doc: "A custom admonition in Tohoku purple with a question mark icon.",
  arg: { type: String, doc: "The title of the admonition." },
  options: {
    collapsed: { type: Boolean, doc: "Whether to collapse the admonition." },
  },
  body: { type: String, doc: "The body of the directive." },
  run(data, vfile, ctx) {
    const admonition = {
      type: "admonition",
      kind: "custom", // instead of "tip"
      class: "dropdown tohokupurple",
      icon: "question", // question mark icon
      children: [
        {
          type: "admonitionTitle",
          children: ctx.parseMyst(data.arg.trim()).children[0].children,
        },
        {
          type: "paragraph",
          children: ctx.parseMyst(data.body.trim()).children[0].children,
        },
      ],
    };
    return [admonition];
  },
};

const plugin = {
  name: "Tohoku Purple Admonition",
  directives: [myAdmonition],
};

// Inject CSS dynamically so no external file is needed
function injectStyle() {
  if (typeof document !== "undefined") {
    const styleId = "tohoku-admonition-style";
    if (!document.getElementById(styleId)) {
      const style = document.createElement("style");
      style.id = styleId;
      style.textContent = `
        .admonition.tohokupurple {
          border-color: #7B1FA2;
        }
        .admonition.tohokupurple > .admonition-title {
          background-color: #7B1FA2;
          color: #fff;
        }
        .admonition.tohokupurple .admonition-title:before {
          content: "?";
          margin-right: 0.5em;
          font-weight: bold;
        }
      `;
      document.head.appendChild(style);
    }
  }
}

// 注入样式
injectStyle();

export default plugin;
