/* ============================================================
   VIDIO Frontend — Application Bootstrap
   Registers routes and initializes the SPA.
   ============================================================ */

// --- Register all routes ---
Router.register('/login',          pageLogin);
Router.register('/dashboard',      pageDashboard);
Router.register('/patients',       pagePatients);
Router.register('/patients/:id',   pagePatientDetail);
Router.register('/studies',        pageStudies);
Router.register('/studies/:id',    pageStudyDetail);
Router.register('/series/:id',     pageSeriesDetail);
Router.register('/images',         pageImages);
Router.register('/images/:id',     pageImageDetail);
Router.register('/upload',         pageUpload);
Router.register('/analysis',       pageAnalysis);
Router.register('/findings',       pageFindings);
Router.register('/findings/:id',   pageFindingDetail);
Router.register('/processes',      pageProcesses);
Router.register('/analytics',      pageAnalyticsDashboard);
Router.register('/models',         pageModels);
Router.register('/users',          pageUsers);

// --- Boot ---
Router.init();
