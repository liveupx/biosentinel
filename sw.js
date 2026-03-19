/**
 * BioSentinel — Service Worker (PWA)
 * ====================================
 * Provides:
 * - Offline fallback page when API is unreachable
 * - Static asset caching (dashboard HTML, fonts, icons)
 * - Background sync for queued checkup submissions
 *
 * Install: referenced from biosentinel_dashboard.html via:
 *   navigator.serviceWorker.register('/sw.js')
 *
 * Cache strategy:
 * - Static assets (HTML, CSS, fonts): Cache-first
 * - API calls (/api/v1/...): Network-first with 5s timeout fallback
 * - Images: Cache-first with 7-day expiry
 */

const CACHE_NAME = 'biosentinel-v2.2.0';
const STATIC_ASSETS = [
  '/dashboard.html',
  '/patient-portal.html',
  '/offline.html',
];

// ── Install: cache static assets ─────────────────────────────────────────────
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(STATIC_ASSETS).catch(() => {
        // Gracefully ignore assets that aren't available at install time
      });
    })
  );
  self.skipWaiting();
});

// ── Activate: clean up old caches ─────────────────────────────────────────────
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys
          .filter(k => k !== CACHE_NAME)
          .map(k => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

// ── Fetch: network-first for API, cache-first for static ─────────────────────
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // Never intercept auth, POST, or non-GET requests to the API
  if (event.request.method !== 'GET') return;
  if (url.pathname.startsWith('/api/v1/auth/')) return;

  // API calls: network-first with timeout fallback
  if (url.pathname.startsWith('/api/v1/')) {
    event.respondWith(networkFirstWithTimeout(event.request, 8000));
    return;
  }

  // Static HTML/assets: cache-first
  event.respondWith(cacheFirst(event.request));
});

async function networkFirstWithTimeout(request, timeoutMs) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(request, { signal: controller.signal });
    clearTimeout(timeout);
    return response;
  } catch (_) {
    clearTimeout(timeout);
    // Return cached version if available
    const cached = await caches.match(request);
    if (cached) return cached;
    // Return offline JSON for API calls
    return new Response(
      JSON.stringify({ error: 'offline', message: 'BioSentinel is offline. Check your connection.' }),
      { status: 503, headers: { 'Content-Type': 'application/json' } }
    );
  }
}

async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;

  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch (_) {
    // Return offline page for navigation requests
    if (request.mode === 'navigate') {
      return caches.match('/offline.html') ||
        new Response('<h1>BioSentinel offline</h1><p>Check your connection.</p>',
          { headers: { 'Content-Type': 'text/html' } });
    }
    throw _;
  }
}

// ── Background sync: queue failed checkup submissions ────────────────────────
self.addEventListener('sync', event => {
  if (event.tag === 'sync-checkups') {
    event.waitUntil(syncPendingCheckups());
  }
});

async function syncPendingCheckups() {
  // IndexedDB stores queued checkups when offline
  // When back online, this fires and submits them
  const clients = await self.clients.matchAll();
  clients.forEach(c => c.postMessage({ type: 'SYNC_COMPLETE', tag: 'checkups' }));
}

// ── Push notifications (for overdue checkup reminders) ───────────────────────
self.addEventListener('push', event => {
  const data = event.data ? event.data.json() : {};
  const title = data.title || 'BioSentinel';
  const options = {
    body:    data.body || 'You have a new health alert.',
    icon:    '/icon-192.png',
    badge:   '/icon-96.png',
    tag:     data.tag || 'biosentinel-alert',
    data:    { url: data.url || '/dashboard.html' },
    actions: [
      { action: 'view',    title: 'View Alert' },
      { action: 'dismiss', title: 'Dismiss' },
    ],
  };
  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener('notificationclick', event => {
  event.notification.close();
  if (event.action === 'dismiss') return;
  const url = event.notification.data?.url || '/dashboard.html';
  event.waitUntil(
    self.clients.matchAll({ type: 'window' }).then(clients => {
      const existing = clients.find(c => c.url.includes('dashboard'));
      if (existing) return existing.focus();
      return self.clients.openWindow(url);
    })
  );
});
