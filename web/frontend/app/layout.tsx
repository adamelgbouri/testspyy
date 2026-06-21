import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Commodity Trading Desk",
  description: "Supply & demand, term structure, risk and options for commodity desks.",
  icons: { icon: "/favicon.ico" },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
