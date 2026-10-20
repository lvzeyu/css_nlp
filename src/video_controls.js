(() => {
  // The MyST book theme renders .mp4 images as <video autoplay muted loop> without
  // playback controls. Turn the controls on for every video on the page.
  function enableControls() {
    document.querySelectorAll('video:not([data-css-nlp-controls])').forEach((video) => {
      video.setAttribute('data-css-nlp-controls', '');
      video.controls = true;
    });
  }

  function start() {
    enableControls();
    new MutationObserver(enableControls).observe(document.body, {
      childList: true,
      subtree: true,
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start, { once: true });
  } else {
    start();
  }
})();
