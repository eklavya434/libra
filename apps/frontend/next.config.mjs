/** @type {import('next').NextConfig} */
const nextConfig = {
  output: process.env.VERCEL ? undefined : "standalone",
  reactStrictMode: true,
  poweredByHeader: false,
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: false,
  },
  async rewrites() {
    const defaultBackend =
      process.env.NODE_ENV === "development"
        ? "http://127.0.0.1:8000"
        : "https://libra-backend-yijf.onrender.com";
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || defaultBackend;
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
