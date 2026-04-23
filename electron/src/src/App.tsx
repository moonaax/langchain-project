import { useState, useRef, useCallback, useEffect } from 'react';
import { ConfigProvider, theme, App as AntApp } from 'antd';
import { Bubble, Sender, Conversations } from '@ant-design/x';
import XMarkdown from '@ant-design/x-markdown';
import type { ConversationsProps } from '@ant-design/x';

const API = 'http://127.0.0.1:8000';

// ── 类型定义 ──

type ContentBlock =
  | { type: 'text'; text: string }
  | { type: 'tool_start'; tool: string; input: string }
  | { type: 'tool_end'; tool: string; output: string };

interface Msg {
  key: string;
  role: 'user' | 'ai';
  content: string;
  streaming?: boolean;
  blocks: ContentBlock[];
}

// ── SSE 解析 ──

interface SSEEvent {
  type: 'tool_start' | 'tool_end' | 'token' | 'done';
  data?: any;
}

function parseSSEEvent(block: string): SSEEvent | null {
  const lines = block.split('\n');
  let evtType = '', evtData = '';
  for (const l of lines) {
    if (l.startsWith('event: ')) evtType = l.slice(7);
    else if (l.startsWith('data: ')) evtData = l.slice(6);
  }
  if (!evtType && !evtData) return null;
  if (evtType === 'tool_start') return { type: 'tool_start', data: JSON.parse(evtData) };
  if (evtType === 'tool_end') return { type: 'tool_end', data: JSON.parse(evtData) };
  if (evtData === '[DONE]') return { type: 'done' };
  if (evtData) return { type: 'token', data: evtData };
  return null;
}

// ── 思考指示器 ──

function ThinkingIndicator({ v }: { v: typeof darkVars }) {
  return (
    <div style={{ display: 'flex', gap: 5, padding: '4px 0' }}>
      {[0, 1, 2].map(i => (
        <span key={i} style={{
          width: 6, height: 6, borderRadius: '50%', background: v.text3,
          animation: `dotBounce 1.2s ${i * 0.15}s infinite ease-in-out`,
        }} />
      ))}
      <style>{`
        @keyframes dotBounce {
          0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
          40% { transform: scale(1); opacity: 1; }
        }
      `}</style>
    </div>
  );
}

// ── 工具卡片 ──

function ToolCard({ block, allBlocks, index, v }: {
  block: ContentBlock & { type: 'tool_start' };
  allBlocks: ContentBlock[];
  index: number;
  v: typeof darkVars;
}) {
  const [expanded, setExpanded] = useState(false);
  const endBlock = allBlocks.slice(index + 1).find(
    b => b.type === 'tool_end' && b.tool === block.tool
  ) as (ContentBlock & { type: 'tool_end' }) | undefined;
  const isDone = !!endBlock;

  return (
    <div style={{
      margin: '6px 0', borderRadius: 10, fontSize: 12, overflow: 'hidden',
      background: v.glass, border: `1px solid ${v.border}`,
      backdropFilter: 'blur(20px)',
      transition: 'border-color 0.3s',
      borderColor: isDone ? 'rgba(34,197,94,0.25)' : 'rgba(249,115,22,0.2)',
    }}>
      {/* Header */}
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '8px 12px', cursor: 'pointer',
          background: isDone ? 'rgba(34,197,94,0.04)' : 'rgba(249,115,22,0.04)',
          transition: 'background 0.2s',
        }}
      >
        <span style={{ fontSize: 14 }}>{isDone ? '✅' : '⏳'}</span>
        <span style={{ fontWeight: 600, color: v.text2 }}>{block.tool}</span>
        <span style={{ flex: 1 }} />
        <span style={{
          fontSize: 10, color: isDone ? 'rgba(34,197,94,0.8)' : 'rgba(249,115,22,0.8)',
          fontWeight: 500,
        }}>
          {isDone ? 'Done' : 'Running...'}
        </span>
        <span style={{
          fontSize: 10, color: v.text3, transition: 'transform 0.2s',
          transform: expanded ? 'rotate(180deg)' : 'rotate(0)',
        }}>▾</span>
      </div>

      {/* Body */}
      {expanded && (
        <div style={{ borderTop: `1px solid ${v.border}` }}>
          <div style={{ padding: '8px 12px', background: 'rgba(0,0,0,0.08)' }}>
            <div style={{ fontSize: 10, color: v.text3, marginBottom: 4, fontWeight: 600 }}>Input</div>
            <div style={{
              fontFamily: "'SF Mono',Menlo,monospace", fontSize: 11, color: v.text2,
              whiteSpace: 'pre-wrap', wordBreak: 'break-all',
            }}>{block.input}</div>
          </div>
          {endBlock && (
            <div style={{ padding: '8px 12px' }}>
              <div style={{ fontSize: 10, color: v.text3, marginBottom: 4, fontWeight: 600 }}>Output</div>
              <div style={{ fontSize: 12, maxHeight: 200, overflow: 'auto' }}>
                <XMarkdown>{renderMarkdown(endBlock.output)}</XMarkdown>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Markdown 渲染 ──

// 修复一个被 || 压缩的表格段
// 处理两种情况：
// 1. || 在行内：| cell1 | cell2 ||| cell3 | cell4 || → 多行表格
// 2. || 在行尾：| header1 | header2 || → 后面跟着数据行（由调用方拼接）
function fixOneCollapsedTable(tableText: string): string {
  const rowSegs = tableText.split(/\|{2,}/).filter(s => s.trim());
  const rawRows: string[][] = [];
  for (const seg of rowSegs) {
    const inner = seg.replace(/^\||\|$/g, '');
    const cells = inner.split('|').map(c => c.trim()).filter(Boolean);
    if (cells.length >= 2) rawRows.push(cells);
  }
  if (rawRows.length < 2) return tableText;

  // 过滤分隔行和残缺行
  const dataRows = rawRows.filter(r => {
    const nonEmpty = r.filter(c => c.trim());
    return nonEmpty.length >= 2 && !r.every(c => /^[\s\-:]+$/.test(c));
  });
  if (dataRows.length < 2) return tableText;

  const maxCols = Math.max(...dataRows.map(r => r.filter(c => c.trim()).length));
  const normalized = dataRows.map(r => {
    const nonEmpty = r.filter(c => c.trim());
    while (nonEmpty.length < maxCols) nonEmpty.push('');
    return `| ${nonEmpty.join(' | ')} |`;
  });
  normalized.splice(1, 0, `| ${Array(maxCols).fill('---').join(' | ')} |`);
  return normalized.join('\n');
}

// 当 tableSeg 没有 || 时（表头行的 || 被剥离），用单 | 分割构建表格
function buildTableFromPipes(text: string): string {
  const rows = text.split('\n').filter(l => l.trim());
  const parsed = rows.map(r => {
    const inner = r.replace(/^\||\|$/g, '');
    return inner.split('|').map(c => c.trim()).filter(Boolean);
  }).filter(r => r.length >= 2);
  if (parsed.length < 2) return text;
  const maxCols = Math.max(...parsed.map(r => r.length));
  const normalized = parsed.map(r => {
    while (r.length < maxCols) r.push('');
    return `| ${r.join(' | ')} |`;
  });
  normalized.splice(1, 0, `| ${Array(maxCols).fill('---').join(' | ')} |`);
  return normalized.join('\n');
}

// 修复文本中所有被 || 压缩的表格（支持多个表格）
// 策略：循环查找 ||，处理一个替换一个，用占位符防止重复处理
function fixAllCollapsedTables(text: string): string {
  let result = text;
  const placeholders: string[] = [];
  const safety = 30;

  for (let iter = 0; iter < safety; iter++) {
    const pos = result.indexOf('||');
    if (pos === -1) break;

    // 向左扩展：找到行首
    let start = pos;
    while (start > 0 && result[start - 1] !== '\n') start--;

    // 向右扩展：找到行尾
    let end = pos + 2;
    while (end < result.length && result[end] !== '\n') end++;

    // 向上查找表头行：如果上一行有 |，且当前行以 | 开头，上一行是表头
    if (start > 0) {
      let prevEnd = start - 1;
      while (prevEnd > 0 && result[prevEnd] !== '\n') prevEnd--;
      const prevLine = result.slice(prevEnd, start - 1);
      if (prevLine.includes('|') && result[start] === '|') {
        start = prevEnd;
      }
    }

    // 向下查找数据行：如果当前行包含 || 且下一行以 | 开头，扩展到包含数据行
    const currentLine = result.slice(start, end);
    if (currentLine.includes('||')) {
      let nextLineEnd = end;
      if (end < result.length && result[end] === '\n') {
        let nextStart = end + 1;
        while (nextStart < result.length && result[nextStart] !== '\n') nextStart++;
        const nextLine = result.slice(end + 1, nextStart);
        if (nextLine.startsWith('|')) {
          nextLineEnd = nextStart;
          // 继续向下扩展更多数据行
          while (nextLineEnd < result.length) {
            if (result[nextLineEnd] !== '\n') break;
            let checkStart = nextLineEnd + 1;
            while (checkStart < result.length && result[checkStart] !== '\n') checkStart++;
            const checkLine = result.slice(nextLineEnd + 1, checkStart);
            if (!checkLine.startsWith('|')) break;
            nextLineEnd = checkStart;
          }
          end = nextLineEnd;
        }
      }
    }

    const block = result.slice(start, end);
    if (!block.includes('||')) continue;

    // 分离前缀文本和表格内容
    const firstPipe = block.indexOf('|');
    let prefix = '';
    let tableContent = block;
    if (firstPipe > 0) {
      const before = block.slice(0, firstPipe);
      if (!before.endsWith('|')) {
        prefix = before.trim();
        tableContent = block.slice(firstPipe);
      }
    }

    // 去掉末尾的 ||（行终止符），同时收集后续以 | 开头的数据行
    let trailingDataLines = '';
    if (tableContent.trimEnd().endsWith('||')) {
      tableContent = tableContent.trimEnd().slice(0, -2);
      // 表头的 || 被去掉后，收集紧跟的数据行
      let lookFrom = end;
      while (lookFrom < result.length) {
        if (result[lookFrom] === '\n') {
          let lineStart = lookFrom + 1;
          while (lineStart < result.length && result[lineStart] !== '\n') lineStart++;
          const line = result.slice(lookFrom + 1, lineStart);
          if (line.startsWith('|')) {
            trailingDataLines += '\n' + line;
            lookFrom = lineStart;
          } else {
            break;
          }
        } else {
          break;
        }
      }
    }

    const fullTable = tableContent + trailingDataLines;
    if (!fullTable.includes('||')) {
      // 没有 || 了，用单 | 分割构建表格
      const fixed = buildTableFromPipes(fullTable);
      const idx = placeholders.length;
      placeholders.push((prefix ? prefix + '\n' : '') + fixed + '\n');
    } else {
      const fixed = fixOneCollapsedTable(fullTable);
      const idx = placeholders.length;
      placeholders.push((prefix ? prefix + '\n' : '') + fixed + '\n');
    }
    result = result.slice(0, start) + `__TBL_${placeholders.length}__` + result.slice(end);
  }

  for (let i = placeholders.length - 1; i >= 0; i--) {
    result = result.replace(`__TBL_${i + 1}__`, placeholders[i]);
  }
  return result;
}

function renderMarkdown(text: string) {
  // 1. 保护 fenced code blocks
  const codeBlocks: string[] = [];
  let raw = text.replace(/```[\s\S]*?```/g, (m) => {
    codeBlocks.push(m);
    return `__CODE_BLOCK_${codeBlocks.length - 1}__`;
  });
  // 2. 保护 inline code（防止 | 被误判为表格分隔符）
  const inlineCodes: string[] = [];
  raw = raw.replace(/`[^`]+`/g, (m) => {
    inlineCodes.push(m);
    return `__INLINE_CODE_${inlineCodes.length - 1}__`;
  });
  // 3. 修复被 || 压缩的表格
  raw = fixAllCollapsedTables(raw);
  // 4. 格式修正
  raw = raw
    .replace(/([^\n])(#{1,6}\s)/g, '$1\n\n$2')
    .replace(/([^\n])(```)/g, '$1\n\n$2')
    .replace(/([^\n])(- \*\*)/g, '$1\n$2')
    .replace(/([^\n])(\d+\.\s)/g, '$1\n$2')
    .replace(/\n{2,}(\|)/g, '\n$1');
  // 5. 恢复 inline code 和 fenced code blocks
  let finalMd = raw.replace(/__INLINE_CODE_(\d+)__/g, (_, i) => inlineCodes[+i]);
  finalMd = finalMd.replace(/__CODE_BLOCK_(\d+)__/g, (_, i) => codeBlocks[+i]);
  const fenceCount = (finalMd.match(/```/g) || []).length;
  if (fenceCount % 2 !== 0) finalMd += '\n```';
  return finalMd;
}

// ── 主题变量 ──

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

// ── 主组件 ──

export default function App() {
  const [messagesMap, setMessagesMap] = useState<Record<string, Msg[]>>({ default: [] });
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [isDark, setIsDark] = useState(true);
  const [conversations, setConversations] = useState<ConversationsProps['items']>([
    { key: 'default', label: 'New conversation' },
  ]);
  const [activeConv, setActiveConv] = useState('default');
  const [useLangGraph, setUseLangGraph] = useState(false);
  const [modeHint, setModeHint] = useState('');
  const idRef = useRef(0);
  const v = isDark ? darkVars : lightVars;

  const messages = messagesMap[activeConv] || [];

  // 模式切换提示
  const toggleMode = useCallback(() => {
    const next = !useLangGraph;
    setUseLangGraph(next);
    const currentMsgs = messagesMap[activeConv] || [];
    if (currentMsgs.length > 0) {
      setModeHint(next ? 'New messages will use LangGraph mode' : 'New messages will use AgentExecutor mode');
      setTimeout(() => setModeHint(''), 3000);
    }
  }, [useLangGraph, messagesMap, activeConv]);

  const updateMessages = useCallback((sessionId: string, updater: (prev: Msg[]) => Msg[]) => {
    setMessagesMap(prev => ({
      ...prev,
      [sessionId]: updater(prev[sessionId] || []),
    }));
  }, []);

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || loading) return;
    setInput('');
    setLoading(true);
    const sid = activeConv;
    const userKey = `msg-${idRef.current++}`;
    const aiKey = `msg-${idRef.current++}`;

    updateMessages(sid, prev => [
      ...prev,
      { key: userKey, role: 'user', content: text, blocks: [] },
      { key: aiKey, role: 'ai', content: '', streaming: true, blocks: [] },
    ]);
    setConversations(prev =>
      prev?.map(c => c.key === sid && c.label === 'New conversation'
        ? { ...c, label: text.slice(0, 30) } : c)
    );

    try {
      const endpoint = useLangGraph ? '/graph_chat' : '/chat';
      const res = await fetch(`${API}${endpoint}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, session_id: sid }),
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
          const evt = parseSSEEvent(part);
          if (!evt) continue;

          if (evt.type === 'tool_start') {
            const inputStr = typeof evt.data.input === 'object' ? JSON.stringify(evt.data.input) : String(evt.data.input);
            updateMessages(sid, prev => prev.map(m => m.key === aiKey ? {
              ...m, blocks: [...m.blocks, { type: 'tool_start', tool: evt.data.tool, input: inputStr }],
            } : m));
          } else if (evt.type === 'tool_end') {
            updateMessages(sid, prev => prev.map(m => m.key === aiKey ? {
              ...m, blocks: [...m.blocks, { type: 'tool_end', tool: evt.data.tool, output: evt.data.output }],
            } : m));
          } else if (evt.type === 'token') {
            updateMessages(sid, prev => prev.map(m => {
              if (m.key !== aiKey) return m;
              const newContent = m.content + evt.data;
              const blocks = [...m.blocks];
              const last = blocks[blocks.length - 1];
              if (last && last.type === 'text') {
                blocks[blocks.length - 1] = { ...last, text: last.text + evt.data };
              } else {
                blocks.push({ type: 'text', text: evt.data });
              }
              return { ...m, content: newContent, blocks };
            }));
          }
        }
      }
    } catch (e: any) {
      updateMessages(sid, prev => prev.map(m =>
        m.key === aiKey ? { ...m, content: `Connection error: ${e.message}`, blocks: [{ type: 'text', text: `Connection error: ${e.message}` }] } : m));
    }
    updateMessages(sid, prev => prev.map(m =>
      m.key === aiKey ? { ...m, streaming: false } : m));
    setLoading(false);
  }, [loading, activeConv, useLangGraph, updateMessages]);

  const clearChat = useCallback(async () => {
    await fetch(`${API}/clear`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: activeConv }),
    }).catch(() => {});
    updateMessages(activeConv, () => []);
    setConversations(prev => prev?.map(c =>
      c.key === activeConv ? { ...c, label: 'New conversation' } : c));
  }, [activeConv, updateMessages]);

  const newChat = useCallback(() => {
    const key = `conv-${Date.now()}`;
    setConversations(prev => [...(prev || []), { key, label: 'New conversation' }]);
    setMessagesMap(prev => ({ ...prev, [key]: [] }));
    setActiveConv(key);
  }, []);

  /* ── 渲染 AI 消息 ── */
  const renderBlocks = (msg: Msg) => (
    <div>
      {msg.streaming && msg.blocks.length === 0 && <ThinkingIndicator v={v} />}
      {(msg.blocks || []).map((block, i) => {
        if (block.type === 'tool_start') {
          return <ToolCard key={i} block={block} allBlocks={msg.blocks} index={i} v={v} />;
        }
        if (block.type === 'tool_end') return null;
        if (block.type === 'text') {
          const md = renderMarkdown(block.text);
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
            <button onClick={newChat} style={{
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
                {useLangGraph ? 'LangGraph' : 'AgentExecutor'}
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
              <button onClick={toggleMode} style={{
                fontSize: 11, padding: '5px 12px', borderRadius: 8, cursor: 'pointer',
                background: useLangGraph ? 'rgba(249,115,22,0.15)' : v.glass,
                border: `1px solid ${useLangGraph ? 'rgba(249,115,22,0.3)' : v.border}`,
                color: useLangGraph ? '#f97316' : v.text3,
                fontFamily: 'inherit', WebkitAppRegion: 'no-drag' as any,
                transition: 'all 0.2s',
              }}>{useLangGraph ? '🔗 LangGraph' : '⚡ Agent'}</button>
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
                  <div style={{ fontSize: 13, color: v.text3 }}>
                    {useLangGraph ? '🔗 LangGraph ReAct Mode' : '⚡ AgentExecutor Mode'} · Powered by DeepSeek
                  </div>
                </div>
              ) : (
                <Bubble.List
                  autoScroll
                  style={{ flex: 1, minHeight: 0 }}
                  role={roles}
                  items={messages.map(m => ({
                    key: m.key,
                    role: m.role,
                    content: m.role === 'user' ? m.content : ((m.blocks?.length ?? 0) > 0 || m.streaming ? m.content : ''),
                    streaming: m.streaming,
                    loading: m.role === 'ai' && m.streaming && (m.blocks?.length ?? 0) === 0,
                  }))}
                />
              )}
            </div>

            {/* 模式切换提示 */}
            {modeHint && (
              <div style={{
                textAlign: 'center', fontSize: 12, color: '#f97316', padding: '4px 0',
                background: 'rgba(249,115,22,0.06)',
              }}>{modeHint}</div>
            )}

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
