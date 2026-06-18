declare module "react-plotly.js/factory" {
  import * as React from "react";
  import type { PlotParams } from "react-plotly.js";

  /** Returns a React component that uses the given Plotly module. */
  export default function createPlotlyComponent(
    Plotly: unknown
  ): React.ComponentType<PlotParams>;
}

declare module "plotly.js-dist-min" {
  const Plotly: any;
  export default Plotly;
}
