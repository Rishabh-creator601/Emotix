import React from "react";
import { createRoot } from "react-dom/client";
import App, { ErrorBoundary } from "./App.jsx";
import "./styles.css";

const rootElement = document.getElementById("root");
window.addEventListener("error", (event) => {
  rootElement.innerHTML = `<main class="boot-screen"><h1>Emotix</h1><p>${event.message}</p></main>`;
});

window.addEventListener("unhandledrejection", (event) => {
  rootElement.innerHTML = `<main class="boot-screen"><h1>Emotix</h1><p>${event.reason?.message || event.reason}</p></main>`;
});

createRoot(rootElement).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>,
);
