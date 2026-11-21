/* Keep diagram viewing on the chapter, including after instant navigation. */
(() => {
  let viewer;
  let image;
  let trigger;
  let previousOverflow;
  let previousPosition;
  let previousTop;
  let scrollY;
  let previousWidth;
  let navigating = false;

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
    const error = document.createElement('p');
    error.textContent = 'The diagram could not be loaded. Close this view and try again.';
    error.hidden = true;
    image.addEventListener('error', () => { error.hidden = !viewer.open; });
    image.addEventListener('load', () => { error.hidden = true; });
    viewer.append(close, image, error);
    document.body.append(viewer);
    close.addEventListener('click', () => viewer.close());
    viewer.addEventListener('click', (event) => {
      if (event.target === viewer) viewer.close();
    });
    viewer.addEventListener('close', () => {
      document.body.style.overflow = previousOverflow;
      document.body.style.position = previousPosition;
      document.body.style.top = previousTop;
      document.body.style.width = previousWidth;
      if (!navigating) window.scrollTo(0, scrollY);
      image.removeAttribute('src');
      if (!navigating && trigger?.isConnected) trigger.focus({ preventScroll: true });
    });
  }

  const closeViewer = () => { if (viewer?.open) { navigating = true; viewer.close(); } };
  window.addEventListener('popstate', closeViewer);
  window.addEventListener('pagehide', closeViewer);
  if (typeof document$ !== 'undefined') document$.subscribe(closeViewer);

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
    navigating = false;
    trigger = link;
    image.src = url.href;
    image.alt = thumbnail.alt || 'Diagram';
    viewer.setAttribute('aria-label', image.alt);
    previousOverflow = document.body.style.overflow;
    previousPosition = document.body.style.position;
    previousTop = document.body.style.top;
    scrollY = window.scrollY;
    previousWidth = document.body.style.width;
    document.body.style.width = '100%';
    document.body.style.position = 'fixed';
    document.body.style.top = `-${scrollY}px`;
    document.body.style.overflow = 'hidden';
    viewer.showModal();
  }, true);
})();
