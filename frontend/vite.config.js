import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The backend allows requests from this port (see backend/app/config.py, PEDICXR_CORS).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true },
});
