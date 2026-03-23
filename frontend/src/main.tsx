import { FormEvent, useState } from 'react'
import { createRoot } from 'react-dom/client'

type Source = { filename: string; chunk_index: number; score: number }
type Judge = { score: number; reasoning: string }
type Debug = { sub_queries: string[]; chunks_retrieved: number; top_scores: { file: string; score: number }[]; episodic_memories_used: number; judge?: Judge; refinements?: number }
type Message = { role: 'user' | 'assistant'; content: string; sources?: Source[]; debug?: Debug }

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
  const [logs, setLogs] = useState<string[]>([])
  const [processing, setProcessing] = useState(false)
  const [input, setInput] = useState('')
  const [thinking, setThinking] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])

  const processFfu = async () => {
    setProcessing(true)
    setStatus('Processing...')
    setLogs([])
    try {
      const res = await fetch('/api/process', { method: 'POST' })
      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop()!
        for (const part of parts) {
          if (!part.startsWith('data: ')) continue
          const data = JSON.parse(part.slice(6))
          if (data.type === 'log') setLogs((p) => [...p, data.msg])
          else if (data.type === 'done') setStatus(`Done: ${data.documents} documents, ${data.chunks} chunks indexed`)
          else if (data.type === 'error') setStatus(`Error: ${data.error}`)
        }
      }
    } catch (e) {
      setStatus(`Error: ${e}`)
    }
    setProcessing(false)
  }

  const [statusMsg, setStatusMsg] = useState('')

  const send = async (e: FormEvent) => {
    e.preventDefault()
    if (!input.trim() || thinking) return
    const history = messages.map(({ role, content }) => ({ role, content }))
    const question = input.trim()
    setInput('')
    setThinking(true)
    setStatusMsg('Starting...')
    setMessages((m) => [...m, { role: 'user', content: question }])

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: question, history }),
      })
      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop()!
        for (const part of parts) {
          if (!part.startsWith('data: ')) continue
          const data = JSON.parse(part.slice(6))
          if (data.type === 'status') {
            setStatusMsg(data.msg)
          } else if (data.type === 'done') {
            setMessages((m) => [
              ...m,
              { role: 'assistant', content: data.response, sources: data.sources, debug: data.debug },
            ])
          } else if (data.type === 'error') {
            setMessages((m) => [
              ...m,
              { role: 'assistant', content: `Error: ${data.error}` },
            ])
          }
        }
      }
    } catch (e) {
      setMessages((m) => [
        ...m,
        { role: 'assistant', content: `Error: ${e}` },
      ])
    }
    setThinking(false)
    setStatusMsg('')
  }

  return (
    <div style={ui.page}>
      <div style={ui.app}>
        <button onClick={processFfu} disabled={processing} style={ui.field}>
          {processing ? 'Processing...' : 'Process FFU'}
        </button>
        <div>{status}</div>
        {logs.length > 0 && (
          <div style={{ maxHeight: 140, overflowY: 'auto', padding: 8, background: '#1a1a2e', color: '#0f0', fontSize: 12, fontFamily: 'monospace', borderRadius: 4 }}>
            {logs.map((l, i) => <div key={i}>{l}</div>)}
          </div>
        )}
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
              {message.debug && (
                <div style={{ marginTop: 6, padding: 8, background: '#1a1a2e', color: '#a0a0a0', fontSize: 11, fontFamily: 'monospace', borderRadius: 4, maxWidth: '80%' }}>
                  <div style={{ color: '#6ee7b7' }}>Queries: [{message.debug.sub_queries.join(' | ')}]</div>
                  <div>Retrieved {message.debug.chunks_retrieved} chunks — top: {message.debug.top_scores.map(s => `${s.file.slice(0, 25)}(${s.score})`).join(', ')}</div>
                  <div style={{ color: message.debug.episodic_memories_used > 0 ? '#fbbf24' : '#666' }}>
                    Episodic memories used: {message.debug.episodic_memories_used}
                  </div>
                  {message.debug.refinements != null && message.debug.refinements > 0 && (
                    <div style={{ color: '#c084fc' }}>Refined {message.debug.refinements}x after judge feedback</div>
                  )}
                  {message.debug.judge && message.debug.judge.score >= 0 && (
                    <div style={{ color: message.debug.judge.score >= 0.7 ? '#6ee7b7' : message.debug.judge.score >= 0.4 ? '#fbbf24' : '#f87171' }}>
                      Judge: {message.debug.judge.score}/1.0 — {message.debug.judge.reasoning}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
          {thinking && <div style={{ ...ui.msg, ...ui.assistant, color: '#666' }}>{statusMsg || 'Thinking...'}</div>}
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
