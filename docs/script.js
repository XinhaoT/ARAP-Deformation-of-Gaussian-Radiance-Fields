"use strict";

const copyButton = document.querySelector("[data-copy-bibtex]");
const bibtexCode = document.querySelector("#bibtex-code");
const copyStatus = document.querySelector("#copy-status");

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textArea = document.createElement("textarea");
  textArea.value = text;
  textArea.setAttribute("readonly", "");
  textArea.style.position = "fixed";
  textArea.style.opacity = "0";
  document.body.appendChild(textArea);
  textArea.select();
  const copied = document.execCommand("copy");
  textArea.remove();

  if (!copied) {
    throw new Error("Copy command failed");
  }
}

if (copyButton && bibtexCode && copyStatus) {
  copyButton.addEventListener("click", async () => {
    try {
      await copyText(bibtexCode.textContent.trim());
      copyButton.textContent = "Copied";
      copyStatus.textContent = "Citation copied to clipboard.";
    } catch {
      copyButton.textContent = "Select citation";
      copyStatus.textContent = "Copy was unavailable. Select the citation text manually.";
    }

    window.setTimeout(() => {
      copyButton.textContent = "Copy citation";
      copyStatus.textContent = "";
    }, 2400);
  });
}
