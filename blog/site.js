// Navigation chrome shared by the blog posts and the project pages: breadcrumb, byline,
// table of contents and previous/next links. Posts are listed here, newest first.
(function () {
    var POSTS = [
        ['conformal_sphere.html', 'Conformal Maps to the Sphere from Laplace Eigenfunctions', 'November 2026'],
        ['exact_predicates.html', 'Adaptive Exact Predicates in Distance Space', 'October 2026'],
        ['more_delaunay.html', 'Even More Delaunay Triangulation', 'September 2025'],
        ['edge.html', 'Constrained Delaunay Edge Insertion', 'July 2024'],
        ['delaunay.html', 'Coordinate-free Delaunay Triangulation', 'January 2024'],
        ['geodesic.html', 'Distance Geometry Processing and Geodesic Paths', 'December 2023']
    ];

    var h1 = document.querySelector('h1');
    if (!h1) return;

    function el(tag, cls, html) {
        var e = document.createElement(tag);
        if (cls) e.className = cls;
        if (html) e.innerHTML = html;
        return e;
    }

    var file = location.pathname.split('/').pop();
    var inBlog = location.pathname.indexOf('/blog/') !== -1;
    var idx = -1;
    POSTS.forEach(function (p, i) { if (p[0] === file) idx = i; });

    // Breadcrumb
    var crumbs = el('div', 'crumbs', inBlog
        ? '<a href="index.html">&larr; blog</a>'
        : '<a href="index.html">&larr; nick nebel</a>');
    h1.parentNode.insertBefore(crumbs, h1);

    if (idx < 0) return;

    // Byline
    h1.parentNode.insertBefore(el('div', 'byline', POSTS[idx][2]), h1.nextSibling);

    // Previous / next
    var nav = el('nav', 'post-nav');
    var newer = POSTS[idx - 1], older = POSTS[idx + 1];
    nav.innerHTML =
        (newer ? '<a class="newer" href="' + newer[0] + '"><span>&larr; Newer</span>' + newer[1] + '</a>' : '<span></span>') +
        (older ? '<a class="older" href="' + older[0] + '"><span>Older &rarr;</span>' + older[1] + '</a>' : '<span></span>');
    document.body.appendChild(nav);

    // Table of contents from the top-level section headings
    var heads = Array.prototype.filter.call(document.querySelectorAll('h2'), function (h) {
        return !h.closest('.digression');
    });
    if (heads.length < 3) return;

    var used = {};
    var items = heads.map(function (h) {
        var id = h.id || h.textContent.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'section';
        while (used[id]) id += '-';
        used[id] = true;
        h.id = id;
        return '<li><a href="#' + id + '">' + h.textContent + '</a></li>';
    }).join('');

    var inline = el('details', 'toc-inline', '<summary>Contents</summary><ol>' + items + '</ol>');
    var side = el('nav', 'toc-side', '<div class="toc-title">Contents</div><ol>' + items + '</ol>');
    h1.nextSibling.parentNode.insertBefore(inline, h1.nextSibling.nextSibling);
    document.body.appendChild(side);

    // Highlight the section being read
    if ('IntersectionObserver' in window) {
        var links = side.querySelectorAll('a');
        var obs = new IntersectionObserver(function (entries) {
            entries.forEach(function (e) {
                if (!e.isIntersecting) return;
                links.forEach(function (a) {
                    a.classList.toggle('current', a.getAttribute('href') === '#' + e.target.id);
                });
            });
        }, { rootMargin: '0px 0px -70% 0px' });
        heads.forEach(function (h) { obs.observe(h); });
    }
})();
