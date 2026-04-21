import { useState, useRef, useCallback } from 'react';
import { ConfigProvider, theme, App as AntApp } from 'antd';
import { Bubble, Sender, Conversations } from '@ant-design/x';
import XMarkdown from '@ant-design/x-markdown';
import type { ConversationsProps } from '@ant-design/x';

const API = 'http://127.0.0.1:8000';

// 每条消息的内容块：文本或工具调用，按时间顺序排列
type ContentBlock =
  | { type: 'text'; text: string }
  | { type: 'tool_start'; tool: string; input: string }
  | { type: 'tool_end'; tool: string; output: string };

interface Msg {
  key: string;
  role: 'user' | 'ai';
  content: string;       // 纯文本累积（给 Bubble 用）
  streaming?: boolean;
  blocks: ContentBlock[]; // 按时间顺序的内容块
}

const darkVars = {
  bg: '#07070a', sidebar: 'rgba(12,12,18,0.5)', glass: 'rgba(255,255,255,0.035)',
  glassHover: 'rgba(255,255,255,0.065)', border: 'rgba(255,255,255,0.08)',
  text: '#f2f2f7', text2: '#b0b0c0', text3: '#70708a',
};
const lightVars = {
  bg: '#f5f5f7', sidebar: 'rgba(255,255,255,0.6)', glass: 'rgba(255,255,255,0.5)',
  glassHover: 'rgba(255,255,255,0.7)', border: 'rgba(0,0,0,0.08)',
  text: '#1a1a1e', text2: '#52525b', text3: '#8a8a96',
};

export default function App() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [isDark, setIsDark] = useState(true);
  const [conversations, setConversations] = useState<ConversationsProps['items']>([
    { key: 'default', label: 'New conversation' },
  ]);
  const [activeConv, setActiveConv] = useState('default');
  const idRef = useRef(0);
  const v = isDark ? darkVars : lightVars;

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || loading) return;
    setInput('');
    setLoading(true);
    const userKey = `msg-${idRef.current++}`;
    const aiKey = `msg-${idRef.current++}`;
    setMessages(prev => [
      ...prev,
      { key: userKey, role: 'user', content: text, blocks: [] },
      { key: aiKey, role: 'ai', content: '', streaming: true, blocks: [] },
    ]);
    setConversations(prev =>
      prev?.map(c => c.key === activeConv && c.label === 'New conversation'
        ? { ...c, label: text.slice(0, 30) } : c)
    );
    try {
      const res = await fetch(`${API}/chat`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });
      const reader = res.body!.getReader();
      const dec = new TextDecoder();
      let buf = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const parts = buf.split('\n\n');
        buf = parts.pop()!;
        for (const part of parts) {
          if (!part.trim()) continue;
          const lines = part.split('\n');
          let evtType = '', evtData = '';
          for (const l of lines) {
            if (l.startsWith('event: ')) evtType = l.slice(7);
            else if (l.startsWith('data: ')) evtData = l.slice(6);
          }
          if (evtType === 'tool_start') {
            const info = JSON.parse(evtData);
            const inputStr = typeof info.input === 'object' ? JSON.stringify(info.input) : String(info.input);
            setMessages(prev => prev.map(m => m.key === aiKey ? {
              ...m, blocks: [...m.blocks, { type: 'tool_start', tool: info.tool, input: inputStr }],
            } : m));
          } else if (evtType === 'tool_end') {
            const info = JSON.parse(evtData);
            setMessages(prev => prev.map(m => m.key === aiKey ? {
              ...m, blocks: [...m.blocks, { type: 'tool_end', tool: info.tool, output: info.output }],
            } : m));
          } else if (evtData && evtData !== '[DONE]') {
            console.log('[SSE token]', JSON.stringify(evtData));
            setMessages(prev => prev.map(m => {
              if (m.key !== aiKey) return m;
              const newContent = m.content + evtData;
              // 追加到最后一个 text block，或新建一个
              const blocks = [...m.blocks];
              const last = blocks[blocks.length - 1];
              if (last && last.type === 'text') {
                blocks[blocks.length - 1] = { ...last, text: last.text + evtData };
              } else {
                blocks.push({ type: 'text', text: evtData });
              }
              return { ...m, content: newContent, blocks };
            }));
          }
        }
      }
    } catch (e: any) {
      setMessages(prev => prev.map(m =>
        m.key === aiKey ? { ...m, content: `Connection error: ${e.message}`, blocks: [{ type: 'text', text: `Connection error: ${e.message}` }] } : m));
    }
    setMessages(prev => prev.map(m =>
      m.key === aiKey ? { ...m, streaming: false } : m));
    setLoading(false);
  }, [loading, activeConv]);

  const clearChat = useCallback(async () => {
    await fetch(`${API}/clear`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: 'default' }),
    }).catch(() => {});
    setMessages([]);
    setConversations(prev => prev?.map(c =>
      c.key === activeConv ? { ...c, label: 'New conversation' } : c));
  }, [activeConv]);

  /* ── 渲染 AI 消息的内容块 ── */
  const renderBlocks = (msg: Msg) => (
    <div>
      {(msg.blocks || []).map((block, i) => {
        if (block.type === 'tool_start') {
          // 找对应的 tool_end
          const endBlock = (msg.blocks || []).slice(i + 1).find(
            b => b.type === 'tool_end' && b.tool === block.tool
          ) as (ContentBlock & { type: 'tool_end' }) | undefined;
          return (
            <details key={i} style={{
              marginBottom: 8, padding: '8px 12px', fontSize: 12,
              background: v.glass, border: `1px solid ${v.border}`,
              borderRadius: 8, backdropFilter: 'blur(20px)',
            }}>
              <summary style={{ cursor: 'pointer', color: '#f97316', fontWeight: 600 }}>
                🔧 调用 {block.tool} {endBlock ? '✅' : '⏳'}
              </summary>
              <div style={{ padding: '4px 0', color: v.text3, fontFamily: "'SF Mono',Menlo,monospace", fontSize: 11 }}>{block.input}</div>
              {endBlock && <div style={{
                padding: '4px 0', color: v.text2, fontSize: 11,
                maxHeight: 80, overflow: 'auto', borderTop: `1px solid ${v.border}`, marginTop: 4,
              }}>{endBlock.output}</div>}
            </details>
          );
        }
        if (block.type === 'tool_end') return null; // 已在 tool_start 中渲染
        if (block.type === 'text') {
          // 先把代码块提取出来保护，避免内部 # 被当标题
          const codeBlocks: string[] = [];
          let raw = block.text.replace(/```[\s\S]*?```/g, (m) => {
            codeBlocks.push(m);
            return `__CODE_BLOCK_${codeBlocks.length - 1}__`;
          });
          raw = raw
            .replace(/([^\n])(#{1,6}\s)/g, '$1\n\n$2')
            .replace(/([^\n])(```)/g, '$1\n\n$2')
            .replace(/([^\n])(- \*\*)/g, '$1\n$2')
            .replace(/([^\n])(\d+\.\s)/g, '$1\n$2')
            // 表格修复：在 | 开头的单元格前插入换行
            .replace(/ \| (?=[^|\n]*\|)/g, ' |\n| ')
            .replace(/\| \|/g, '|\n|')
            .replace(/\n{2,}(\|)/g, '\n$1')
            .replace(/([^\n])(\|[-:]+)/g, '$1\n$2')
            .replace(/([^\n])(\| )/g, '$1\n$2');
          // 还原代码块
          let finalMd = raw.replace(/__CODE_BLOCK_(\d+)__/g, (_, i) => codeBlocks[+i]);
          // 流式过程中代码块可能未闭合，补上 ```
          const fenceCount = (finalMd.match(/```/g) || []).length;
          if (fenceCount % 2 !== 0) finalMd += '\n```';
          const md = finalMd;
          console.log('[Markdown raw]', JSON.stringify(block.text.slice(0, 300)));
          console.log('[Markdown processed]', JSON.stringify(md.slice(0, 300)));
          return <XMarkdown key={i}>{md}</XMarkdown>;
        }
        return null;
      })}
    </div>
  );

  const roles: Bubble.ListProps['role'] = {
    user: { placement: 'end', variant: 'shadow' },
    ai: {
      placement: 'start',
      variant: 'outlined',
      avatar: (
        <div style={{
          width: 28, height: 28,
          background: 'linear-gradient(135deg, #f97316, #fb923c, #fbbf24)',
          borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 14, boxShadow: '0 0 20px rgba(249,115,22,0.3)',
        }}>⚡</div>
      ),
      contentRender: (_, info) => {
        const msg = messages.find(m => m.key === info?.key);
        if (!msg) return null;
        return renderBlocks(msg);
      },
    },
  };

  return (
    <ConfigProvider theme={{
      algorithm: isDark ? theme.darkAlgorithm : theme.defaultAlgorithm,
      token: {
        colorPrimary: '#f97316',
        colorBgContainer: isDark ? '#0d0d12' : '#ffffff',
        colorBgElevated: isDark ? '#12121a' : '#ffffff',
        fontFamily: "'Inter', -apple-system, sans-serif",
      },
    }}>
      <AntApp>
        <style>{`
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
          @keyframes float16 { to { transform: translate(60px, 40px) } }
          @keyframes float14 { to { transform: translate(-50px, -30px) } }
          @keyframes float18 { to { transform: translate(-30px, 50px) } }
          body { background: ${v.bg} !important; transition: background 0.3s; }
          .ant-conversations .ant-conversations-item { border-radius: 10px !important; }
          .ant-sender { border-radius: 12px !important; backdrop-filter: blur(40px) saturate(1.4); box-shadow: 0 4px 30px ${isDark ? 'rgba(0,0,0,0.2)' : 'rgba(0,0,0,0.06)'}; transition: all 0.3s !important; }
          .ant-sender:focus-within { box-shadow: 0 0 50px rgba(249,115,22,0.06), 0 0 100px rgba(168,85,247,0.04), 0 8px 32px ${isDark ? 'rgba(0,0,0,0.2)' : 'rgba(0,0,0,0.06)'} !important; border-color: ${isDark ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.14)'} !important; }
          table { width: 100%; border-collapse: collapse; display: block; overflow-x: auto; font-size: 13px; }
          th, td { border: 1px solid ${v.border}; padding: 6px 10px; white-space: nowrap; }
          th { background: ${v.glass}; font-weight: 600; }
        `}</style>

        {/* 环境光晕 */}
        <div style={{ position: 'fixed', width: 500, height: 500, borderRadius: '50%', background: `radial-gradient(circle, ${isDark ? 'rgba(168,85,247,0.13)' : 'rgba(168,85,247,0.08)'}, transparent 70%)`, filter: 'blur(100px)', pointerEvents: 'none', zIndex: 0, top: '-15%', left: '-8%' }} />
        <div style={{ position: 'fixed', width: 450, height: 450, borderRadius: '50%', background: `radial-gradient(circle, ${isDark ? 'rgba(99,102,241,0.11)' : 'rgba(99,102,241,0.06)'}, transparent 70%)`, filter: 'blur(100px)', pointerEvents: 'none', zIndex: 0, bottom: '-10%', right: '-5%' }} />

        <div style={{ display: 'flex', height: '100vh', position: 'relative', zIndex: 1 }}>
          {/* ── 侧边栏 ── */}
          <div style={{
            width: 250, display: 'flex', flexDirection: 'column',
            background: v.sidebar, backdropFilter: 'blur(60px) saturate(1.6)',
            borderRight: `1px solid ${v.border}`,
          }}>
            <div style={{ padding: '52px 16px 14px', WebkitAppRegion: 'drag' as any }}>
              <div style={{ fontSize: 14, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 9, color: v.text }}>
                <span style={{
                  width: 26, height: 26, borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 13,
                  background: 'linear-gradient(135deg, #f97316, #fb923c, #fbbf24)',
                  boxShadow: '0 0 20px rgba(249,115,22,0.3)',
                }}>⚡</span>
                DeepSeek Chat
              </div>
            </div>
            <button onClick={clearChat} style={{
              margin: '6px 12px 4px', padding: '9px 0', fontSize: 13, fontFamily: 'inherit',
              background: v.glass, border: `1px solid ${v.border}`, borderRadius: 10,
              color: v.text2, cursor: 'pointer', WebkitAppRegion: 'no-drag' as any,
            }}>+ New chat</button>
            <div style={{ flex: 1, overflow: 'auto', padding: '4px 8px' }}>
              <Conversations items={conversations} activeKey={activeConv} onActiveChange={setActiveConv} />
            </div>
            <div style={{
              padding: '12px 16px', borderTop: `1px solid ${v.border}`,
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            }}>
              <span style={{ fontSize: 11, color: v.text3, display: 'flex', alignItems: 'center', gap: 7 }}>
                <span style={{ width: 5, height: 5, borderRadius: '50%', background: '#22c55e', boxShadow: '0 0 8px rgba(34,197,94,0.6)' }} />
                deepseek-chat
              </span>
              <button onClick={() => setIsDark(!isDark)} style={{
                width: 30, height: 30, borderRadius: 8, fontSize: 14, cursor: 'pointer',
                background: v.glass, border: `1px solid ${v.border}`, color: v.text3,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                WebkitAppRegion: 'no-drag' as any,
              }}>{isDark ? '🌙' : '☀️'}</button>
            </div>
          </div>

          {/* ── 主区域 ── */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            {/* 顶栏 */}
            <div style={{
              height: 44, paddingTop: 28, boxSizing: 'content-box',
              borderBottom: `1px solid ${v.border}`, display: 'flex', alignItems: 'center',
              paddingLeft: 20, paddingRight: 16,
              background: isDark ? 'rgba(7,7,10,0.4)' : 'rgba(245,245,247,0.5)',
              backdropFilter: 'blur(40px)', WebkitAppRegion: 'drag' as any, flexShrink: 0,
            }}>
              <span style={{ fontSize: 13, color: v.text3, fontWeight: 500 }}>
                {conversations?.find(c => c.key === activeConv)?.label}
              </span>
              <span style={{ flex: 1 }} />
              <button onClick={clearChat} style={{
                fontSize: 11, padding: '5px 12px', borderRadius: 8, cursor: 'pointer',
                background: v.glass, border: `1px solid ${v.border}`, color: v.text3,
                fontFamily: 'inherit', WebkitAppRegion: 'no-drag' as any,
              }}>Clear</button>
            </div>

            {/* 消息区域 */}
            <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
              {messages.length === 0 ? (
                <div style={{
                  flex: 1, display: 'flex', flexDirection: 'column',
                  alignItems: 'center', justifyContent: 'center', gap: 8, paddingBottom: 80,
                }}>
                  <div style={{
                    width: 64, height: 64, borderRadius: 18, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 28, marginBottom: 12,
                    background: 'linear-gradient(135deg, #f97316, #fb923c, #fbbf24)',
                    boxShadow: '0 0 50px rgba(249,115,22,0.3), 0 0 100px rgba(249,115,22,0.1)',
                  }}>⚡</div>
                  <div style={{ fontSize: 22, fontWeight: 700, color: v.text }}>What can I help with?</div>
                  <div style={{ fontSize: 13, color: v.text3 }}>Powered by LangChain + DeepSeek</div>
                </div>
              ) : (
                <Bubble.List
                  autoScroll
                  style={{ flex: 1, minHeight: 0 }}
                  role={roles}
                  items={messages.map(m => ({
                    key: m.key,
                    role: m.role,
                    content: m.role === 'user' ? m.content : ((m.blocks?.length ?? 0) > 0 ? m.content : ''),
                    streaming: m.streaming,
                    loading: m.role === 'ai' && m.streaming && (m.blocks?.length ?? 0) === 0,
                  }))}
                />
              )}
            </div>

            {/* 输入框 */}
            <div style={{ padding: '0 24px 24px', flexShrink: 0 }}>
              <Sender
                value={input}
                onChange={setInput}
                onSubmit={sendMessage}
                loading={loading}
                placeholder="Ask anything..."
              />
              <div style={{ textAlign: 'center', fontSize: 11, color: v.text3, opacity: 0.6, marginTop: 8 }}>
                ⏎ Send · ⇧⏎ New line
              </div>
            </div>
          </div>
        </div>
      </AntApp>
    </ConfigProvider>
  );
}
