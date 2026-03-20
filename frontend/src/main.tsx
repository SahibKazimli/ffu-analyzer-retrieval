import { FormEvent, useState } from 'react'
import { createRoot } from 'react-dom/client'

type Source = { filename: string; chunk_index: number; score: number }
type Message = { role: 'user' | 'assistant'; content: string; sources?: Source[] }

const ui = {
  page: { margin: 0, minHeight: '100vh', background: '#f2f2f2', color: '#222', fontFamily: 'system-ui, sans-serif' } as const,
  app: { maxWidth: 800, margin: '0 auto', padding: 24, display: 'grid', gap: 12 } as const,
  chat: { minHeight: 360, padding: 12, border: '1px solid #ddd', background: '#fafafa', overflow: 'auto', display: 'grid', gap: 8, alignContent: 'start' } as const,
  form: { display: 'grid', gridTemplateColumns: '1fr auto', gap: 8 } as const,
  field: { padding: '10px 12px', border: '1px solid #ccc', background: '#fff', font: 'inherit' } as const,
  msg: { maxWidth: '80%', padding: '10px 12px', borderRadius: 8, whiteSpace: 'pre-wrap' as const },
  user: { justifySelf: 'end' as const, background: '#dcdcdc' },
  assistant: { justifySelf: 'start' as const, background: '#ededed' },
  sources: { display: 'flex', flexWrap: 'wrap' as const, gap: 4, marginTop: 6 },
  sourceTag: {
    display: 'inline-block',
    padding: '2px 8px',
    background: '#e0e7ff',
    color: '#3730a3',
    borderRadius: 4,
    fontSize: 11,
    fontWeight: 500,
  } as const,
}

function App() {
  const [status, setStatus] = useState('')
  const [input, setInput] = useState('')
  const [thinking, setThinking] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])

  const processFfu = async () => {
    setStatus('Processing (extracting, chunking, embedding)...')
    try {
      const data = await fetch('/api/process', { method: 'POST' }).then((r) => r.json())
      setStatus(`Done: ${data.documents} documents, ${data.chunks} chunks indexed`)
    } catch (e) {
      setStatus(`Error: ${e}`)
    }
  }

  const send = async (e: FormEvent) => {
    e.preventDefault()
    if (!input.trim() || thinking) return
    const history = messages.map(({ role, content }) => ({ role, content }))
    const question = input.trim()
    setInput('')
    setThinking(true)
    setMessages((m) => [...m, { role: 'user', content: question }])

    try {
      const data = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: question, history }),
      }).then((r) => r.json())

      setMessages((m) => [
        ...m,
        { role: 'assistant', content: data.response, sources: data.sources },
      ])
    } catch (e) {
      setMessages((m) => [
        ...m,
        { role: 'assistant', content: `Error: ${e}` },
      ])
    }
    setThinking(false)
  }

  return (
    <div style={ui.page}>
      <div style={ui.app}>
        <button onClick={processFfu} style={ui.field}>Process FFU</button>
        <div>{status}</div>
        <div style={ui.chat}>
          {messages.map((message, i) => (
            <div key={i}>
              <div style={{ ...ui.msg, ...(message.role === 'user' ? ui.user : ui.assistant) }}>
                {message.content}
              </div>
              {message.sources && message.sources.length > 0 && (
                <div style={{ ...ui.sources, justifySelf: 'start' }}>
                  {message.sources.map((s, j) => (
                    <span key={j} style={ui.sourceTag} title={`Score: ${s.score}`}>
                      📄 {s.filename}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
          {thinking && <div style={{ ...ui.msg, ...ui.assistant, color: '#666' }}>Thinking...</div>}
        </div>
        <form onSubmit={send} style={ui.form}>
          <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask about the FFU documents" style={ui.field} />
          <button style={ui.field}>Send</button>
        </form>
      </div>
    </div>
  )
}

createRoot(document.getElementById('root')!).render(<App />)
