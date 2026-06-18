/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: { typedRoutes: true },
  webpack: (config) => {
    // react-plotly.js's default entry imports `plotly.js/dist/plotly`,
    // which isn't shipped — we redirect it to the lighter minified dist.
    config.resolve.alias = {
      ...(config.resolve.alias || {}),
      "plotly.js/dist/plotly": "plotly.js-dist-min",
    };
    return config;
  },
};

export default nextConfig;
