import { useRef, useState } from "react";

// Shows the X-ray with the Grad-CAM heatmap on top. The heatmap PNG from the backend has
// the same size as the X-ray, and both use object-fit: contain in the same box, so they line up exactly.
export default function XrayViewer({ imageUrl, heatmapUrl }) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [showHeat, setShowHeat] = useState(true);
  const [opacity, setOpacity] = useState(55);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [invert, setInvert] = useState(false);
  const [loadError, setLoadError] = useState(false);
  const start = useRef(null);

  const setZoomTo = (z) => {
    const nz = Math.min(4, Math.max(1, Math.round(z * 100) / 100));
    setZoom(nz);
    if (nz === 1) setPan({ x: 0, y: 0 });
  };
  const reset = () => {
    setZoomTo(1); setBrightness(100); setContrast(100); setInvert(false); setShowHeat(true); setOpacity(55);
  };

  const onPointerDown = (e) => {
    if (zoom === 1) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    start.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
    setDragging(true);
  };
  const onPointerMove = (e) => {
    if (!start.current) return;
    setPan({ x: e.clientX - start.current.x, y: e.clientY - start.current.y });
  };
  const endDrag = () => { start.current = null; setDragging(false); };

  return (
    <div className="stagewrap">
      <div
        className={`stage${zoom > 1 ? " pannable" : ""}${dragging ? " panning" : ""}`}
        onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={endDrag} onPointerCancel={endDrag}
      >
        {loadError ? (
          <span className="stage-msg">The X-ray image could not be loaded. Check that the backend is running.</span>
        ) : (
          <div className="zoomer" style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` }}>
            <img
              src={imageUrl} alt="Chest X-ray" draggable={false}
              style={{ filter: `brightness(${brightness}%) contrast(${contrast}%)${invert ? " invert(1)" : ""}` }}
              onError={() => setLoadError(true)}
            />
            {heatmapUrl && (
              <img className="heat" src={heatmapUrl} alt="" draggable={false} style={{ opacity: showHeat ? opacity / 100 : 0 }} />
            )}
          </div>
        )}
        <span className="stage-tag">R</span>
        <span className="stage-tag r">L</span>
        <span className="stage-tag b">{Math.round(zoom * 100)}%</span>
      </div>
      <div className="vtools">
        <div className="line">
          <button className="btn sm" type="button" aria-label="Zoom out" onClick={() => setZoomTo(zoom / 1.25)}>−</button>
          <button className="btn sm" type="button" aria-label="Zoom in" onClick={() => setZoomTo(zoom * 1.25)}>+</button>
          <button className="btn sm" type="button" onClick={reset}>Reset</button>
          {heatmapUrl && (
            <label className="switch"><input id="tHeat" type="checkbox" checked={showHeat} onChange={(e) => setShowHeat(e.target.checked)} />Grad-CAM heatmap</label>
          )}
          <label className="switch"><input id="tInv" type="checkbox" checked={invert} onChange={(e) => setInvert(e.target.checked)} />Invert</label>
        </div>
        <div className="line">
          {heatmapUrl && (
            <label htmlFor="rOp">Heatmap <input id="rOp" type="range" min="0" max="100" value={opacity} onChange={(e) => setOpacity(+e.target.value)} /></label>
          )}
          <label htmlFor="rBr">Brightness <input id="rBr" type="range" min="50" max="160" value={brightness} onChange={(e) => setBrightness(+e.target.value)} /></label>
          <label htmlFor="rCo">Contrast <input id="rCo" type="range" min="50" max="180" value={contrast} onChange={(e) => setContrast(+e.target.value)} /></label>
        </div>
        <span className="help">When zoomed in, drag the X-ray to move it.</span>
      </div>
    </div>
  );
}
