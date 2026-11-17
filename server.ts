import 'dotenv/config';
import http from 'http';
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, Type } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const httpServer = http.createServer(app);
const PORT = Number(process.env.PORT) || 3000;

app.use(express.json({ limit: '2mb' }));

let cachedAi: GoogleGenAI | null = null;
function getAiClient(): GoogleGenAI {
  if (!cachedAi) {
    cachedAi = new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  }
  return cachedAi;
}

app.get('/api/health', (_req, res) => {
  res.json({ ok: true });
});

const RESPONSE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    thoughtSteps: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description: '3-4 шага размышлений именно над тем, что написал пользователь.',
    },
    reply: {
      type: Type.STRING,
      description: 'Прямой, умный и естественный ответ пользователю.',
    },
    artifactTitle: {
      type: Type.STRING,
      description: 'Название файла артефакта (например, app.js, plan.md) или пустая строка, если артефакт не нужен.',
    },
    artifactCode: {
      type: Type.STRING,
      description: 'Код или структурированный документ артефакта, либо пустая строка.',
    },
  },
  required: ['thoughtSteps', 'reply', 'artifactTitle', 'artifactCode'],
};

async function callGeminiWithRetry(promptContents: string, systemInstruction: string) {
  // Try high-availability gemini-3.1-flash-lite first, then gemini-4-hydrogen with retry
  const modelsToTry = ['gemini-3.1-flash-lite', 'gemini-4-hydrogen', 'gemini-3.1-flash-lite'];
  let lastError: unknown = null;
  const ai = getAiClient();

  for (let attempt = 0; attempt < modelsToTry.length; attempt++) {
    const model = modelsToTry[attempt];
    try {
      const response = await ai.models.generateContent({
        model,
        contents: promptContents,
        config: {
          systemInstruction,
          responseMimeType: 'application/json',
          responseSchema: RESPONSE_SCHEMA,
        },
      });
      const rawText = (response.text ?? '{}').trim();
      return JSON.parse(rawText);
    } catch (err: unknown) {
      lastError = err;
      const msg = err instanceof Error ? err.message : String(err);
      const isTransient = msg.includes('503') || msg.includes('UNAVAILABLE') || msg.includes('high demand') || msg.includes('429');
      if (!isTransient) {
        throw err;
      }
      await new Promise((r) => setTimeout(r, 350 * (attempt + 1)));
    }
  }
  throw lastError;
}

app.post('/api/fable-chat', async (req, res) => {
  try {
    const {
      message,
      history = [],
      modelName = 'Opus 5.5',
      effort = 'Low',
      mode = 'chat',
      attachments = [],
    } = req.body || {};

    if (!message || typeof message !== 'string') {
      res.status(400).json({ error: 'Empty message' });
      return;
    }

    const historyContext =
      Array.isArray(history) && history.length > 0
        ? '\n\nКонтекст предыдущих сообщений диалога:\n' +
          history
            .slice(-8)
            .map((h: { role: string; text: string }) => `${h.role === 'user' ? 'Пользователь' : 'Claude'}: ${h.text}`)
            .join('\n')
        : '';

    const attachContext =
      Array.isArray(attachments) && attachments.length > 0
        ? `\nПрикреплённые файлы: ${attachments.join(', ')}`
        : '';

    const systemInstruction = `Ты — Claude (${modelName}, уровень размышлений: ${effort}) на максимальном тарифе "Ultra plan".
Твоя главная задача — ТОЧНО понимать, что написал пользователь, и отвечать по существу его вопроса, задачи или реплики.
Правила:
1. Всегда отвечай именно на то, о чём спрашивает или что пишет пользователь (если это "Привет" — тепло и естественно поздоровайся и спроси, чем помочь; если вопрос — дай точный, глубокий и понятный ответ; если просит HTML-страницу, виджет, игру или интерфейс — напиши полный, самодостаточный, рабочий HTML со встроенными <style> и <script> в поле artifactCode и укажи artifactTitle с расширением ".html", например "index.html" или "widget.html", чтобы пользователь мог сразу протестировать его в окне Preview прямо в чате!).
2. Если пользователь сам прислал HTML-код в сообщении (например, чтобы протестировать, проверить или улучшить его), кратко разбери код, подскажи улучшения (если нужно) и обязательно верни готовый рабочий вариант в artifactCode с расширением ".html".
3. Если включён режим "${mode === 'plan' ? 'Ultra Plan (режим планирования)' : 'Chat'}", ${
      mode === 'plan'
        ? 'структурируй ответ как чёткий пошаговый план действий по теме запроса пользователя.'
        : 'отвечай естественно, дружелюбно и содержательно.'
    }
4. В поле thoughtSteps верни 3 коротких, осмысленных шага твоих размышлений над этим конкретным запросом пользователя.
5. Поле artifactCode заполняй ТОЛЬКО тогда, когда пользователь просит код, HTML, интерфейс, игру, таблицу или шаблон (иначе оставь artifactTitle и artifactCode пустыми строками).`;

    const parsed = await callGeminiWithRetry(
      `Сообщение пользователя: ${message}${attachContext}${historyContext}`,
      systemInstruction
    );
    res.json(parsed);
  } catch (error: unknown) {
    const errMsg = error instanceof Error ? error.message : String(error);
    console.error('Error in /api/fable-chat:', errMsg);
    res.status(503).json({ error: errMsg });
  }
});

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        ws: { server: httpServer },
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
