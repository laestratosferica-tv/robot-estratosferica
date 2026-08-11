import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL("https://promodetector.co"),
  title: "PromoDetector | Encuentra ofertas estratosféricas",
  description:
    "Radar de productos, oportunidades y contenidos verificados para tecnología, gaming y cultura digital.",
  icons: {
    icon: "/favicon.png",
    shortcut: "/favicon.png",
  },
  openGraph: {
    title: "PromoDetector",
    description: "Encuentra ofertas estratosféricas.",
    url: "https://promodetector.co",
    siteName: "PromoDetector",
    images: [{ url: "/og.png", width: 1536, height: 910 }],
    locale: "es_CO",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "PromoDetector",
    description: "Encuentra ofertas estratosféricas.",
    images: ["/og.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="es">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
