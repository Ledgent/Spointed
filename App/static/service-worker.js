self.addEventListener('install', event => {
  event.waitUntil(
    caches.open('blrs-cache').then(cache => {
      return cache.addAll([
        '/',
        '/index.html',
        '/static/favicon/rblrs.ico',
        '/static/css/styles.css',
        '/static/js/main.js'
      ]);
    })
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request).then(response => {
      return response || fetch(event.request);
    })
  );
});
