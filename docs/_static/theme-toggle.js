(() => {
  "use strict";

  const storageKey = "theme";
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)");

  function initialTheme() {
    const storedTheme = localStorage.getItem(storageKey);
    if (storedTheme === "light" || storedTheme === "dark") {
      return storedTheme;
    }
    return prefersDark.matches ? "dark" : "light";
  }

  function applyTheme(theme) {
    document.body.dataset.theme = theme;
    localStorage.setItem(storageKey, theme);

    const nextTheme = theme === "dark" ? "light" : "dark";
    document.querySelectorAll(".theme-toggle").forEach((button) => {
      button.setAttribute("aria-label", `Switch to ${nextTheme} mode`);
      button.setAttribute("title", `Switch to ${nextTheme} mode`);
    });
  }

  applyTheme(initialTheme());

  // Furo normally cycles through Auto, Light, and Dark. Capture the click before
  // Furo handles it so the control behaves as a direct Light/Dark switch.
  document.addEventListener(
    "click",
    (event) => {
      const target = event.target;
      if (!(target instanceof Element)) {
        return;
      }

      const button = target.closest(".theme-toggle");
      if (button === null) {
        return;
      }

      event.preventDefault();
      event.stopImmediatePropagation();
      applyTheme(document.body.dataset.theme === "dark" ? "light" : "dark");
    },
    true,
  );

  window.addEventListener("storage", (event) => {
    if (event.key === storageKey && (event.newValue === "light" || event.newValue === "dark")) {
      applyTheme(event.newValue);
    }
  });
})();
