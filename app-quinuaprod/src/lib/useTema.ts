import { useCallback, useEffect, useState } from 'react'
import { aplicarTema, guardarTema, leerTema, type Tema } from './theme'

export function useTema() {
  const [tema, setTemaState] = useState<Tema>(leerTema)

  const setTema = useCallback((nuevo: Tema) => {
    guardarTema(nuevo)
    setTemaState(nuevo)
  }, [])

  useEffect(() => {
    aplicarTema(tema)
    if (tema !== 'sistema') return
    // Con "sistema" se sigue el cambio del SO en vivo.
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = () => aplicarTema('sistema')
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [tema])

  return { tema, setTema }
}
