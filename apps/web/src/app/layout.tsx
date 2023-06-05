import './globals.css'

import { Inter } from 'next/font/google'
import { ClerkProvider } from '@clerk/nextjs'
import clsx from 'clsx'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'Create Next App',
  description: 'Generated by create next app',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <ClerkProvider>
      <html lang="en" className="h-full">
        <body className={clsx('h-full', inter.className)}>{children}</body>
      </html>
    </ClerkProvider>
  )
}
