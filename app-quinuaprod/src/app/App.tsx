import { useState } from 'react'
import { RouterProvider } from 'react-router'
import { AppProviders } from './providers'
import { createAppRouter } from './routes'

export default function App() {
  const [router] = useState(createAppRouter)

  return (
    <AppProviders>
      <RouterProvider router={router} />
    </AppProviders>
  )
}
