"""
DOM serialization script injected into browser pages.

Produces a compact text representation of the page that highlights
interactive elements and visible text content, suitable for LLM consumption.
"""

DOM_SERIALIZATION_JS = r"""
(function(maxChars) {
  "use strict";

  var refCounter = 0;
  var interactiveInView = [];
  var interactiveOffScreen = [];
  var textSegments = [];

  var INTERACTIVE_SELECTORS = [
    "a", "button", "input", "select", "textarea",
    "[role='button']", "[onclick]", "[tabindex]", "[contenteditable]"
  ];

  var viewportHeight = window.innerHeight || document.documentElement.clientHeight;
  var viewportWidth = window.innerWidth || document.documentElement.clientWidth;

  function isInteractive(el) {
    var tag = el.tagName.toLowerCase();
    if (["a", "button", "input", "select", "textarea"].indexOf(tag) !== -1) {
      return true;
    }
    if (el.getAttribute("role") === "button") return true;
    if (el.hasAttribute("onclick")) return true;
    if (el.hasAttribute("tabindex")) return true;
    if (el.getAttribute("contenteditable") === "true") return true;
    return false;
  }

  function isVisible(el) {
    if (typeof el.getBoundingClientRect !== "function") return false;
    var style = window.getComputedStyle(el);
    if (style.display === "none") return false;
    if (style.visibility === "hidden") return false;
    if (parseFloat(style.opacity) === 0) return false;
    var rect = el.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) return false;
    return true;
  }

  function isInViewport(el) {
    var rect = el.getBoundingClientRect();
    return (
      rect.top < viewportHeight &&
      rect.bottom > 0 &&
      rect.left < viewportWidth &&
      rect.right > 0
    );
  }

  function describeElement(el) {
    var tag = el.tagName.toLowerCase();
    var parts = ["<" + tag];

    var attrs = ["type", "name", "placeholder", "value", "href", "role",
                 "aria-label", "title", "alt", "src", "action", "method"];
    for (var i = 0; i < attrs.length; i++) {
      var val = el.getAttribute(attrs[i]);
      if (val !== null && val !== "") {
        var truncated = val.length > 80 ? val.substring(0, 77) + "..." : val;
        parts.push(attrs[i] + '="' + truncated.replace(/"/g, "'") + '"');
      }
    }

    if (el.checked) parts.push("checked");
    if (el.disabled) parts.push("disabled");
    if (el.readOnly) parts.push("readonly");

    var desc = parts.join(" ") + ">";

    // Add visible text for non-input elements
    if (["input", "select", "textarea"].indexOf(tag) === -1) {
      var text = (el.textContent || "").trim();
      if (text.length > 0) {
        var shortText = text.length > 60 ? text.substring(0, 57) + "..." : text;
        desc += shortText;
      }
    }

    desc += "</" + tag + ">";
    return desc;
  }

  function assignRef(el) {
    refCounter++;
    var ref = "e" + refCounter;
    el.setAttribute("data-agent-ref", ref);
    return ref;
  }

  function extractText(node) {
    if (node.nodeType === Node.TEXT_NODE) {
      var t = (node.textContent || "").trim();
      if (t.length > 0) {
        return t;
      }
      return "";
    }
    return "";
  }

  function walk(node) {
    if (!node) return;

    // Skip invisible containers
    if (node.nodeType === Node.ELEMENT_NODE) {
      var tag = node.tagName.toLowerCase();
      // Skip script, style, noscript, svg, hidden elements
      if (["script", "style", "noscript", "svg", "link", "meta"].indexOf(tag) !== -1) {
        return;
      }
      if (!isVisible(node)) {
        return;
      }

      if (isInteractive(node)) {
        var ref = assignRef(node);
        var desc = "[" + ref + "] " + describeElement(node);
        if (isInViewport(node)) {
          interactiveInView.push(desc);
        } else {
          interactiveOffScreen.push(desc);
        }
        // Don't recurse into interactive elements to avoid duplication
        return;
      }
    }

    // Collect text nodes
    if (node.nodeType === Node.TEXT_NODE) {
      var text = extractText(node);
      if (text.length > 0) {
        textSegments.push(text);
      }
      return;
    }

    // Recurse into child nodes
    var children = node.childNodes;
    for (var i = 0; i < children.length; i++) {
      walk(children[i]);
    }

    // Handle Shadow DOM
    if (node.nodeType === Node.ELEMENT_NODE && node.shadowRoot) {
      var shadowChildren = node.shadowRoot.childNodes;
      for (var j = 0; j < shadowChildren.length; j++) {
        walk(shadowChildren[j]);
      }
    }
  }

  // --- Main execution ---
  walk(document.body);

  // Build output with priority: viewport interactive > off-screen interactive > text
  var output = [];
  output.push("Page: " + window.location.href);
  output.push("Title: " + (document.title || "(no title)"));
  output.push("");

  // Interactive elements section
  if (interactiveInView.length > 0 || interactiveOffScreen.length > 0) {
    output.push("[Interactive Elements]");
    for (var a = 0; a < interactiveInView.length; a++) {
      output.push(interactiveInView[a]);
    }
    for (var b = 0; b < interactiveOffScreen.length; b++) {
      output.push(interactiveOffScreen[b]);
    }
    output.push("");
  }

  // Page content section
  if (textSegments.length > 0) {
    output.push("[Page Content]");
    for (var c = 0; c < textSegments.length; c++) {
      output.push(textSegments[c]);
    }
    output.push("");
  }

  // Scroll indicator
  var scrollTop = window.pageYOffset || document.documentElement.scrollTop || 0;
  var scrollHeight = document.documentElement.scrollHeight || document.body.scrollHeight;
  var clientHeight = window.innerHeight || document.documentElement.clientHeight;
  if (scrollHeight > clientHeight) {
    var pagesBelow = ((scrollHeight - scrollTop - clientHeight) / clientHeight).toFixed(1);
    if (parseFloat(pagesBelow) > 0) {
      output.push("|SCROLL| (" + pagesBelow + " pages below)");
    }
  }

  var result = output.join("\n");

  // Truncate to budget
  if (result.length > maxChars) {
    result = result.substring(0, maxChars);
    // Try to cut at last newline for cleanliness
    var lastNL = result.lastIndexOf("\n");
    if (lastNL > maxChars * 0.8) {
      result = result.substring(0, lastNL);
    }
    result += "\n... [truncated at " + maxChars + " chars]";
  }

  return result;
})(arguments[0] || 50000)
""".strip()
