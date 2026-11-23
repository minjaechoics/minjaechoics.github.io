(() => {
  const ownHosts = new Set([
    location.hostname,
    "minjaechoics.github.io",
    "www.minjaechoics.github.io",
  ]);

  function markExternalLinks(scope = document) {
    scope.querySelectorAll("a[href]").forEach((link) => {
      let url;
      try {
        url = new URL(link.getAttribute("href"), location.href);
      } catch (_) {
        return;
      }

      if (
        (url.protocol === "http:" || url.protocol === "https:") &&
        !ownHosts.has(url.hostname)
      ) {
        link.target = "_blank";
        link.rel = "noopener noreferrer";
      }
    });
  }

  markExternalLinks();
  new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeType === Node.ELEMENT_NODE) markExternalLinks(node);
      });
    });
  }).observe(document.body, { childList: true, subtree: true });
})();
