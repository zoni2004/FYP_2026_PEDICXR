import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import "./styles.css";
import Layout from "./components/Layout.jsx";
import { ToastProvider } from "./components/Toast.jsx";
import Worklist from "./pages/Worklist.jsx";
import NewStudy from "./pages/NewStudy.jsx";
import StudyViewer from "./pages/StudyViewer.jsx";
import ModelInfo from "./pages/ModelInfo.jsx";
import NotFound from "./pages/NotFound.jsx";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <ToastProvider>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Navigate to="/worklist" replace />} />
            <Route path="worklist" element={<Worklist />} />
            <Route path="new" element={<NewStudy />} />
            <Route path="studies/:id" element={<StudyViewer />} />
            <Route path="model" element={<ModelInfo />} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </ToastProvider>
    </BrowserRouter>
  </React.StrictMode>
);
