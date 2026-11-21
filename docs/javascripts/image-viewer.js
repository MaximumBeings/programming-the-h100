/* Keep diagram viewing on the chapter, including after instant navigation. */
(() => {
  let viewer;
  let image;
  let trigger;
  let previousOverflow;

  function createViewer() {
    viewer = document.createElement('dialog');
    viewer.className = 'book-image-viewer';
    viewer.setAttribute('aria-label', 'Enlarged diagram');
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'book-image-viewer__close';
    close.textContent = 'Close';
    close.setAttribute('aria-label', 'Close enlarged diagram');
    image = document.createElement('img');
    viewer.append(close, image);
    document.body.append(viewer);
    close.addEventListener('click', () => viewer.close());
    viewer.addEventListener('click', (event) => {
      if (event.target === viewer) viewer.close();
    });
    viewer.addEventListener('close', () => {
      document.body.style.overflow = previousOverflow;
      image.removeAttribute('src');
      if (trigger?.isConnected) trigger.focus({ preventScroll: true });
    });
  }

  document.addEventListener('click', (event) => {
    if (event.defaultPrevented || event.button !== 0 || event.ctrlKey ||
        event.metaKey || event.shiftKey || event.altKey) return;
    const link = event.target.closest?.('.md-content a[href]');
    const thumbnail = link?.querySelector('img');
    if (!thumbnail) return;
    const url = new URL(link.href, document.baseURI);
    if (url.origin !== location.origin || !/\.(svg|png|jpe?g|webp|gif|avif)$/i.test(url.pathname)) return;
    event.preventDefault();
    event.stopPropagation();
    if (!viewer) createViewer();
    trigger = link;
    image.src = url.href;
    image.alt = thumbnail.alt || 'Diagram';
    previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    viewer.showModal();
  }, true);
})();
