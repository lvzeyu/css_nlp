(() => {
  const owner = 'lvzeyu';
  const repository = 'css_nlp';
  const badge = '__COLAB_BADGE_DATA_URL__';

  function updateColabLink() {
    const editLink = document.querySelector('a.myst-fm-edit-link[href*="/edit/"]');
    const existing = document.querySelector('a.css-nlp-colab-link');
    if (!editLink) {
      existing?.remove();
      return;
    }

    const editUrl = new URL(editLink.href);
    const prefix = `/${owner}/${repository}/edit/`;
    const source = editUrl.pathname.startsWith(prefix)
      ? editUrl.pathname.slice(prefix.length)
      : '';
    const branchEnd = source.indexOf('/');
    const branch = source.slice(0, branchEnd);
    const file = source.slice(branchEnd + 1);
    if (!branch || !/^notebook\/[^/]+\.ipynb$/.test(file)) {
      existing?.remove();
      return;
    }

    const colabUrl = `https://colab.research.google.com/github/${owner}/${repository}/blob/${branch}/${file}`;
    const link = existing ?? document.createElement('a');
    if (!existing) {
      link.className = 'css-nlp-colab-link text-inherit hover:text-inherit';
      link.title = 'Open in Google Colab';
      link.setAttribute('aria-label', 'Open in Google Colab');
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.style.cssText = 'display:inline-flex;align-items:center;margin-left:0.5rem';
      const image = document.createElement('img');
      image.src = badge;
      image.alt = '';
      image.setAttribute('aria-hidden', 'true');
      image.style.cssText = 'display:block;width:117px;height:20px;max-width:100%';
      link.append(image);
    }
    link.href = colabUrl;
    const downloads = editLink.parentElement?.querySelector(
      ':scope > .myst-fm-downloads-dropdown'
    );
    const lastControl = downloads ?? editLink;
    if (link.previousElementSibling !== lastControl) lastControl.after(link);
  }

  function start() {
    updateColabLink();
    new MutationObserver(updateColabLink).observe(document.body, {
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
