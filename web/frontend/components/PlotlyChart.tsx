"use client";
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

// react-plotly.js looks for `plotly.js` by default; we redirect it to the
// dist-min build via the factory pattern (smaller bundle, same API).
const Plot = createPlotlyComponent(Plotly as any);

export default Plot;
